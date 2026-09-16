from __future__ import annotations

from chat_rag.rag.agent import AssistantTurn, ToolCall, run_agent, system_prompt
from chat_rag.rag.citations import extract_ids, get_message, resolve_answer


class FakeLLM:
    def __init__(self, turns):
        self.turns = list(turns)
        self.calls = []

    def chat(self, messages, tools):
        self.calls.append({"messages": list(messages), "tools": tools})
        if self.turns:
            return self.turns.pop(0)
        return AssistantTurn(content="fallback", tool_calls=[])


class FakeStreamingLLM(FakeLLM):
    def stream_chat(self, messages, tools, on_token=None):
        turn = self.chat(messages, tools)
        if on_token and turn.content:
            for word in turn.content.split(" "):
                on_token(word + " ")
        return turn


def test_agent_executes_tool_then_answers(ctx):
    target_id = f"{3:016x}"
    llm = FakeLLM(
        [
            AssistantTurn(content=None, tool_calls=[ToolCall("keyword_search", {"query": "dimenticare"})]),
            AssistantTurn(content=f"Te lo dice Alice [{target_id}].", tool_calls=[]),
        ]
    )
    result = run_agent("cosa mi ha detto Alice?", ctx, llm)
    assert result.steps and result.steps[0]["tool"] == "keyword_search"
    assert target_id in result.answer
    assert target_id in result.sources


def test_agent_forced_final_after_max_steps(ctx):
    looping = AssistantTurn(content=None, tool_calls=[ToolCall("list_chats", {})])
    llm = FakeLLM([looping, looping, AssistantTurn(content="Insight senza strumenti.", tool_calls=[])])
    result = run_agent("dimmi qualcosa", ctx, llm, max_steps=2)
    assert result.answer == "Insight senza strumenti."
    # The final call must be made with tools disabled.
    assert llm.calls[-1]["tools"] is None


def test_agent_streams_tokens_and_emits_events(ctx):
    target_id = f"{3:016x}"
    llm = FakeStreamingLLM(
        [
            AssistantTurn(content=None, tool_calls=[ToolCall("keyword_search", {"query": "dimenticare"})]),
            AssistantTurn(content=f"Te lo dice Alice [{target_id}].", tool_calls=[]),
        ]
    )
    tokens: list[str] = []
    events: list[tuple[str, dict]] = []
    result = run_agent(
        "cosa mi ha detto Alice?",
        ctx,
        llm,
        on_token=tokens.append,
        on_event=lambda kind, payload: events.append((kind, payload)),
    )
    assert "".join(tokens).strip() == result.answer.strip()
    kinds = [kind for kind, _ in events]
    assert "tool_call" in kinds and "tool_result" in kinds
    assert any(p["name"] == "keyword_search" for k, p in events if k == "tool_call")


def test_extract_ids():
    text = "come dice [0123456789abcdef] e [abcdef0123456789], punto."
    assert extract_ids(text) == ["0123456789abcdef", "abcdef0123456789"]
    assert extract_ids("nessun id qui") == []


def test_resolve_answer_prefers_cited_ids(raw_conn):
    cited = f"{2:016x}"
    fallback = [f"{1:016x}"]
    cits = resolve_answer(raw_conn, f"risposta [{cited}]", fallback)
    assert [c.id for c in cits] == [cited]
    assert cits[0].found


def test_resolve_answer_falls_back(raw_conn):
    cits = resolve_answer(raw_conn, "nessuna citazione", [f"{1:016x}"])
    assert [c.id for c in cits] == [f"{1:016x}"]


def test_missing_citation_marked(raw_conn):
    cits = resolve_answer(raw_conn, "vedi [ffffffffffffffff]", [])
    assert cits[0].found is False


def test_get_message(raw_conn):
    cit = get_message(raw_conn, f"{1:016x}")
    assert cit is not None
    assert cit.sender == "Alice"
    assert "mare" in cit.text


def test_custom_system_prompt_interpolates(ctx):
    prompt = system_prompt(ctx, base="Oggi {today}. Chat:\n{chats}")
    assert "{today}" not in prompt and "{chats}" not in prompt
    assert "- c (1:1" in prompt
