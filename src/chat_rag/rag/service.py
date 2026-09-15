from __future__ import annotations

from dataclasses import dataclass, field

from ..config import Settings
from ..db.db import open_db
from ..embed.index import chroma_client, ensure_collection
from ..embed.ollama_embed import OllamaEmbedder
from .agent import AgentResult, LLM, OllamaLLM, run_agent
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


def build_context(settings: Settings, collection_name: str = "messages") -> ToolContext:
    conn = open_db(settings.db_path)
    client = chroma_client(settings.chroma_dir)
    collection = ensure_collection(client, collection_name)
    embedder = OllamaEmbedder(settings.embed_model, settings.ollama_host)
    return ToolContext(conn=conn, collection=collection, embedder=embedder)


def answer_question(
    settings: Settings,
    question: str,
    llm: LLM | None = None,
    ctx: ToolContext | None = None,
    max_steps: int = 6,
    system_extra: str | None = None,
    collection_name: str = "messages",
) -> Answer:
    owns_ctx = ctx is None
    if ctx is None:
        settings.ensure_dirs()
        ctx = build_context(settings, collection_name)
    if llm is None:
        llm = OllamaLLM(settings.llm_model, settings.ollama_host)

    result: AgentResult = run_agent(question, ctx, llm, max_steps=max_steps, system_extra=system_extra)
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
