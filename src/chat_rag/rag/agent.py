"""A small hand-rolled tool-calling agent over Ollama (no LangChain).

The agent asks the local LLM to either call one of the tools or answer. Tool
results are fed back until the model produces a final answer (or ``max_steps``
is reached, at which point it is asked to answer with what it has).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol

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


class OllamaLLM:
    def __init__(self, model: str, host: str, temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature
        self._client = ollama.Client(host=host)

    def chat(self, messages: list[dict], tools: list[dict] | None) -> AssistantTurn:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "options": {"temperature": self.temperature},
        }
        if tools:
            kwargs["tools"] = tools
        response = self._client.chat(**kwargs)
        msg = response.message
        calls: list[ToolCall] = []
        for tc in getattr(msg, "tool_calls", None) or []:
            fn = tc.function
            args = fn.arguments
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            calls.append(ToolCall(name=fn.name, arguments=args or {}))
        return AssistantTurn(content=msg.content, tool_calls=calls)


@dataclass
class AgentResult:
    answer: str
    messages: list[dict]
    steps: list[dict]
    sources: dict[str, dict]

    @property
    def fallback_ids(self) -> list[str]:
        return list(self.sources.keys())


def system_prompt(ctx: ToolContext, extra: str | None = None) -> str:
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
- semantic_search = meaning/topics; keyword_search = exact words or nicknames; get_stats = counts/averages/trends; get_context = messages around one id.
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
) -> AgentResult:
    messages: list[dict] = [
        {"role": "system", "content": system_prompt(ctx, system_extra)},
        {"role": "user", "content": question},
    ]
    sources: dict[str, dict] = {}
    steps: list[dict] = []

    for _ in range(max_steps):
        turn = llm.chat(messages, TOOL_SCHEMAS)
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
            result = dispatch(ctx, call.name, call.arguments)
            collect_message_ids(result, sources)
            messages.append({"role": "tool", "content": dumps(result), "tool_name": call.name})
            steps.append({"tool": call.name, "arguments": call.arguments, "result": result})

    messages.append(
        {
            "role": "user",
            "content": "Answer now using only the information gathered, with [id] citations.",
        }
    )
    turn = llm.chat(messages, None)
    return AgentResult(answer=turn.content or "", messages=messages, steps=steps, sources=sources)
