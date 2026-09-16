from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..config import Settings
from ..db.db import open_db
from ..embed.index import chroma_client, ensure_collection
from ..embed.ollama_embed import OllamaEmbedder
from .agent import AgentResult, EventFn, LLM, OllamaLLM, TokenFn, run_agent
from .citations import Citation, resolve_answer
from .tools import ToolContext


@dataclass
class Answer:
    question: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    sources: list[Citation] = field(default_factory=list)
    steps: list[dict] = field(default_factory=list)

    @property
    def tools_used(self) -> list[str]:
        return [s["tool"] for s in self.steps]


def build_context(
    settings: Settings, collection_name: str = "messages", llm: LLM | None = None
) -> ToolContext:
    conn = open_db(settings.db_path)
    client = chroma_client(settings.chroma_dir)
    collection = ensure_collection(client, collection_name)
    embedder = OllamaEmbedder(settings.embed_model, settings.ollama_host)
    return ToolContext(conn=conn, collection=collection, embedder=embedder, llm=llm)


def answer_question(
    settings: Settings,
    question: str,
    llm: LLM | None = None,
    ctx: ToolContext | None = None,
    max_steps: int = 6,
    system_extra: str | None = None,
    system_override: str | None = None,
    collection_name: str = "messages",
    on_event: EventFn | None = None,
    on_token: TokenFn | None = None,
) -> Answer:
    owns_ctx = ctx is None
    if llm is None:
        llm = OllamaLLM(
            settings.llm_model,
            settings.ollama_host,
            num_predict=settings.llm_num_predict,
            think=settings.llm_think,
            temperature=settings.llm_temperature,
        )
    if ctx is None:
        settings.ensure_dirs()
        ctx = build_context(settings, collection_name, llm=llm)
    if system_override is None and settings.system_prompt_file:
        try:
            system_override = Path(settings.system_prompt_file).read_text(encoding="utf-8")
        except OSError:
            system_override = None

    result: AgentResult = run_agent(
        question,
        ctx,
        llm,
        max_steps=max_steps,
        system_extra=system_extra,
        system_override=system_override,
        on_event=on_event,
        on_token=on_token,
    )
    citations = resolve_answer(ctx.conn, result.answer, result.fallback_ids)
    sources = resolve_answer(ctx.conn, "", result.fallback_ids)

    if owns_ctx:
        try:
            ctx.conn.close()
        except Exception:  # noqa: BLE001
            pass

    return Answer(
        question=question,
        answer=result.answer,
        citations=citations,
        sources=sources,
        steps=result.steps,
    )
