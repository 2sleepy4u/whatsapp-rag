"""A small hand-rolled tool-calling agent over Ollama (no LangChain).

The agent asks the local LLM to either call one of the tools or answer. Tool
results are fed back until the model produces a final answer (or ``max_steps``
is reached, at which point it is asked to answer with what it has).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Protocol

import ollama

from .citations import Citation
from .tools import TOOL_SCHEMAS, ToolContext, collect_message_ids, dispatch, dumps


@dataclass
class ToolCall:
    name: str
    arguments: dict


@dataclass
class AssistantTurn:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLM(Protocol):
    def chat(self, messages: list[dict], tools: list[dict] | None) -> AssistantTurn:
        ...


def _parse_tool_calls(raw) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for tc in raw or []:
        fn = tc.function
        args = fn.arguments
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        calls.append(ToolCall(name=fn.name, arguments=args or {}))
    return calls


class OllamaLLM:
    def __init__(
        self,
        model: str,
        host: str,
        temperature: float = 0.0,
        keep_alive: str | int = "30m",
        num_predict: int | None = 1024,
        think: bool | None = False,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.keep_alive = keep_alive
        self.num_predict = num_predict
        self.think = think
        self._client = ollama.Client(host=host)

    def _kwargs(self, messages: list[dict], tools: list[dict] | None, stream: bool) -> dict[str, Any]:
        options: dict[str, Any] = {"temperature": self.temperature}
        if self.num_predict:
            options["num_predict"] = self.num_predict
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "options": options,
            "keep_alive": self.keep_alive,
            "stream": stream,
        }
        if tools:
            kwargs["tools"] = tools
        if self.think is not None:
            kwargs["think"] = self.think
        return kwargs

    def chat(self, messages: list[dict], tools: list[dict] | None) -> AssistantTurn:
        response = self._client.chat(**self._kwargs(messages, tools, stream=False))
        msg = response.message
        return AssistantTurn(content=msg.content, tool_calls=_parse_tool_calls(getattr(msg, "tool_calls", None)))

    def stream_chat(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        on_token: Callable[[str], None] | None = None,
    ) -> AssistantTurn:
        parts: list[str] = []
        calls: list[ToolCall] = []
        for chunk in self._client.chat(**self._kwargs(messages, tools, stream=True)):
            msg = chunk.message
            if msg.content:
                parts.append(msg.content)
                if on_token:
                    on_token(msg.content)
            calls.extend(_parse_tool_calls(getattr(msg, "tool_calls", None)))
        return AssistantTurn(content="".join(parts) or None, tool_calls=calls)


def _invoke(
    llm: LLM,
    messages: list[dict],
    tools: list[dict] | None,
    on_token: Callable[[str], None] | None,
) -> AssistantTurn:
    stream = getattr(llm, "stream_chat", None)
    if stream is not None and on_token is not None:
        return stream(messages, tools, on_token=on_token)
    return llm.chat(messages, tools)


@dataclass
class AgentResult:
    answer: str
    messages: list[dict]
    steps: list[dict]
    sources: dict[str, dict]

    @property
    def fallback_ids(self) -> list[str]:
        return list(self.sources.keys())


EventFn = Callable[[str, dict], None]
TokenFn = Callable[[str], None]


def system_prompt(ctx: ToolContext, extra: str | None = None, base: str | None = None) -> str:
    if base is not None:
        prompt = base
    else:
        try:
            chats = _list_chats_for_prompt(ctx)
        except Exception:  # noqa: BLE001
            chats = ""
        prompt = f"""You are a private, offline analyst for exported WhatsApp conversations.
Today's date: {date.today().isoformat()}.
Available chats:
{chats or '- (none indexed)'}

Rules:
- Always call a tool before answering factual questions. Never invent messages, dates, senders or numbers.
- Prefer hybrid_search (semantic + exact) for factual lookups; window_search for richer context around a discussion; semantic_search = meaning/topics; keyword_search = exact words or nicknames; get_stats = counts/averages/trends; get_context = messages around one id.
- Pass context=2 on searches when the surrounding exchange matters; then call get_context only for deeper digging.
- When you mention or rely on a message, cite its exact id in square brackets, e.g. [3f9a1c2b4d5e6f70].
- Use only ids returned by tools. Do not fabricate ids.
- If the tools return nothing useful, say so honestly instead of guessing.
- Answer in the same language as the user's question (usually Italian). Be concise and concrete.
- Structure: short answer first, then the supporting cited messages."""
    if extra:
        prompt += "\n\n" + extra
    return prompt


def _list_chats_for_prompt(ctx: ToolContext) -> str:
    from .tools import list_chats

    lines = []
    for c in list_chats(ctx)[:20]:
        kind = "group" if c["is_group"] else "1:1"
        lines.append(f"- {c['chat_id']} ({kind}, {c['messages']} msg, {c['first']}..{c['last']})")
    return "\n".join(lines)


def run_agent(
    question: str,
    ctx: ToolContext,
    llm: LLM,
    max_steps: int = 6,
    system_extra: str | None = None,
    system_override: str | None = None,
    on_event: EventFn | None = None,
    on_token: TokenFn | None = None,
) -> AgentResult:
    messages: list[dict] = [
        {"role": "system", "content": system_prompt(ctx, system_extra, system_override)},
        {"role": "user", "content": question},
    ]
    sources: dict[str, dict] = {}
    steps: list[dict] = []

    def emit(kind: str, **payload: Any) -> None:
        if on_event:
            on_event(kind, payload)

    for step_no in range(max_steps):
        emit("step_start", step=step_no + 1, max_steps=max_steps)
        turn = _invoke(llm, messages, TOOL_SCHEMAS, on_token)
        if not turn.tool_calls:
            return AgentResult(answer=turn.content or "", messages=messages, steps=steps, sources=sources)

        messages.append(
            {
                "role": "assistant",
                "content": turn.content or "",
                "tool_calls": [
                    {"function": {"name": c.name, "arguments": c.arguments}} for c in turn.tool_calls
                ],
            }
        )
        for call in turn.tool_calls:
            emit("tool_call", name=call.name, arguments=call.arguments)
            result = dispatch(ctx, call.name, call.arguments)
            collect_message_ids(result, sources)
            count = _result_count(result)
            emit("tool_result", name=call.name, arguments=call.arguments, count=count)
            messages.append({"role": "tool", "content": dumps(result), "tool_name": call.name})
            steps.append({"tool": call.name, "arguments": call.arguments, "result": result})

    messages.append(
        {
            "role": "user",
            "content": "Answer now using only the information gathered, with [id] citations.",
        }
    )
    emit("final", step=max_steps)
    turn = _invoke(llm, messages, None, on_token)
    return AgentResult(answer=turn.content or "", messages=messages, steps=steps, sources=sources)


def _result_count(result: Any) -> int:
    if isinstance(result, list):
        return len(result)
    if isinstance(result, dict):
        for key in ("rows", "messages", "hours"):
            value = result.get(key)
            if isinstance(value, list):
                return len(value)
        if "id" in result and "text" in result:
            return 1
    return 0
