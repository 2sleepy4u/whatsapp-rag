from __future__ import annotations

import json

from chat_rag.config import load_settings
from chat_rag.rag.agent import AssistantTurn
from chat_rag.rag.citations import Citation
from chat_rag.rag.eval import (
    QuestionResult,
    Variant,
    build_variants,
    load_questions,
    run_eval,
    summarize,
)
from chat_rag.rag.service import Answer, answer_question


class FakeLLM:
    def __init__(self, turns):
        self.turns = list(turns)
        self.calls = []

    def chat(self, messages, tools):
        self.calls.append({"messages": list(messages), "tools": tools})
        if self.turns:
            return self.turns.pop(0)
        return AssistantTurn(content="fallback", tool_calls=[])


def _answer(question: str, text: str, found: int, missing: int, tool: str = "semantic_search") -> Answer:
    citations = [Citation(id=f"f{i:016x}", date="", time="", sender="", text="", found=True) for i in range(found)]
    citations += [Citation(id=f"m{i:016x}", date="", time="", sender="", text="", found=False) for i in range(missing)]
    return Answer(question=question, answer=text, citations=citations, steps=[{"tool": tool}])


def test_load_questions_accepts_strings_and_objects(tmp_path):
    path = tmp_path / "q.json"
    path.write_text(json.dumps(["a", {"question": "b"}]), encoding="utf-8")
    assert load_questions(path) == ["a", "b"]


def test_load_questions_rejects_bad_entry(tmp_path):
    path = tmp_path / "q.json"
    path.write_text(json.dumps([{"nope": 1}]), encoding="utf-8")
    try:
        load_questions(path)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_build_variants_cross_product():
    variants = build_variants("PROMPT", [None, True], [0.0, 0.7])
    names = [v.name for v in variants]
    assert names == [
        "base-auto-t0",
        "prompt-auto-t0",
        "base-auto-t0.7",
        "prompt-auto-t0.7",
        "base-think-on-t0",
        "prompt-think-on-t0",
        "base-think-on-t0.7",
        "prompt-think-on-t0.7",
    ]
    assert all(v.temperature == 0.0 for v in variants if v.name.endswith("t0"))
    assert all(v.system_override == "PROMPT" for v in variants if v.name.startswith("prompt"))


def test_build_variants_without_prompt():
    variants = build_variants(None, [False], [0.3])
    assert [v.name for v in variants] == ["base-think-off-t0.3"]
    assert variants[0].temperature == 0.3
    assert variants[0].system_override is None


def test_run_eval_and_summarize():
    variants = [Variant("base"), Variant("prompt")]

    def ask(question, variant):
        if variant.name == "prompt":
            return _answer(question, "ok", found=2, missing=0)
        return _answer(question, "ok", found=1, missing=1)

    results = run_eval(["q1", "q2"], variants, ask)
    assert len(results) == 4
    report = summarize(results)
    assert report["prompt"]["citation_found_rate"] == 1.0
    assert report["base"]["citation_found_rate"] == 0.5
    assert report["base"]["tool_counts"] == {"semantic_search": 2}


def test_run_eval_captures_errors():
    def boom(question, variant):
        raise RuntimeError("nope")

    results = run_eval(["q"], [Variant("x")], boom)
    assert results[0].error == "RuntimeError: nope"
    assert summarize(results)["x"]["errors"] == 1


def test_summarize_handles_no_citations():
    report = summarize([QuestionResult("v", "q", "", [], 0, 0, 0, 0, 1.0)])
    assert report["v"]["citation_found_rate"] is None


def test_answer_question_passes_system_override(ctx):
    settings = load_settings()
    llm = FakeLLM([AssistantTurn(content="ciao", tool_calls=[])])
    ans = answer_question(settings, "domanda", llm=llm, ctx=ctx, system_override="PROMPT PERSONALIZZATO")
    assert ans.answer == "ciao"
    assert llm.calls[0]["messages"][0]["content"] == "PROMPT PERSONALIZZATO"


def test_answer_question_builds_llm_with_temperature(ctx, monkeypatch):
    settings = load_settings()
    captured: dict = {}

    class FakeOllama:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

        def chat(self, messages, tools):
            return AssistantTurn(content="ok", tool_calls=[])

    monkeypatch.setattr("chat_rag.rag.service.OllamaLLM", FakeOllama)
    answer_question(settings, "q", ctx=ctx)
    assert captured["temperature"] == settings.llm_temperature
    assert captured["think"] == settings.llm_think
