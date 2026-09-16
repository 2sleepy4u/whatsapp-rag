from __future__ import annotations

from chat_rag.config import load_settings


def test_llm_think_tristate(monkeypatch):
    monkeypatch.delenv("CHAT_RAG_LLM_THINK", raising=False)
    assert load_settings().llm_think is None
    monkeypatch.setenv("CHAT_RAG_LLM_THINK", "")
    assert load_settings().llm_think is None
    monkeypatch.setenv("CHAT_RAG_LLM_THINK", "1")
    assert load_settings().llm_think is True
    monkeypatch.setenv("CHAT_RAG_LLM_THINK", "0")
    assert load_settings().llm_think is False


def test_embed_context_default(monkeypatch):
    monkeypatch.delenv("CHAT_RAG_EMBED_CONTEXT", raising=False)
    assert load_settings().embed_context == "none"


def test_system_prompt_file_env(monkeypatch):
    monkeypatch.setenv("CHAT_RAG_SYSTEM_PROMPT_FILE", "/tmp/p.txt")
    assert load_settings().system_prompt_file == "/tmp/p.txt"
    monkeypatch.setenv("CHAT_RAG_SYSTEM_PROMPT_FILE", "")
    assert load_settings().system_prompt_file is None
