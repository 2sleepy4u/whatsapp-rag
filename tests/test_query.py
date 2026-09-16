from __future__ import annotations

from chat_rag.rag.agent import AssistantTurn
from chat_rag.rag.query import expand_queries, hyde_document


class FakeLLM:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list = []

    def chat(self, messages, tools):
        self.calls.append((messages, tools))
        return AssistantTurn(content=self.content, tool_calls=[])


def test_expand_queries_cleans_numbers_and_dedupes():
    llm = FakeLLM("1. andiamo al mare\n- mare domenica\nandiamo al mare\nviaggio al mare?")
    assert expand_queries(llm, "mare", n=3) == [
        "andiamo al mare",
        "mare domenica",
        "viaggio al mare?",
    ]


def test_expand_queries_excludes_original_case_insensitive():
    llm = FakeLLM("mare\nMARE\nvai al mare")
    assert expand_queries(llm, "mare", n=3) == ["vai al mare"]


def test_expand_queries_zero_or_error():
    assert expand_queries(FakeLLM("x"), "q", n=0) == []

    class Boom:
        def chat(self, messages, tools):
            raise RuntimeError("nope")

    assert expand_queries(Boom(), "q") == []


def test_expand_queries_uses_no_tools():
    llm = FakeLLM("alternativa")
    expand_queries(llm, "q")
    assert llm.calls[0][1] is None


def test_hyde_document_strips_whitespace():
    assert hyde_document(FakeLLM("  Ci vediamo al mare domenica  "), "quando?") == "Ci vediamo al mare domenica"
