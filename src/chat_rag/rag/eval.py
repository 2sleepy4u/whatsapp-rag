"""Offline evaluation harness for the RAG agent.

Runs a list of questions under one or more *variants* (system prompt / thinking
mode) and aggregates quality signals: tools used, citations that resolved
against the DB, latency and answer length.

This module never talks to Ollama directly. Callers inject an ``ask_fn`` so the
same logic is exercised in tests with fakes.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from .service import Answer

AskFn = Callable[[str, "Variant"], Answer]


@dataclass(frozen=True)
class Variant:
    name: str
    system_override: str | None = None
    think: bool | None = None
    temperature: float = 0.0


@dataclass
class QuestionResult:
    variant: str
    question: str
    answer: str
    tools_used: list[str]
    steps: int
    citations_found: int
    citations_missing: int
    answer_chars: int
    latency_ms: float
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def build_variants(
    prompt_text: str | None,
    thinks: list[bool | None],
    temperatures: list[float],
) -> list[Variant]:
    """Cross-product of prompt/think/temperature into named variants."""
    think_suffix = {None: "-auto", False: "-think-off", True: "-think-on"}
    variants: list[Variant] = []
    for think in thinks:
        for temperature in temperatures:
            suffix = f"{think_suffix[think]}-t{temperature:g}"
            variants.append(
                Variant(name=f"base{suffix}", think=think, temperature=temperature)
            )
            if prompt_text is not None:
                variants.append(
                    Variant(
                        name=f"prompt{suffix}",
                        system_override=prompt_text,
                        think=think,
                        temperature=temperature,
                    )
                )
    return variants


def load_questions(path: str | Path) -> list[str]:
    """Accept a JSON list of strings, or of objects with a ``question`` key."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("questions file must contain a JSON list")
    out: list[str] = []
    for item in data:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict) and item.get("question"):
            out.append(str(item["question"]))
        else:
            raise ValueError(f"invalid question entry: {item!r}")
    return out


def _record(variant: Variant, question: str, answer: Answer, started: float) -> QuestionResult:
    found = sum(1 for c in answer.citations if c.found)
    missing = len(answer.citations) - found
    return QuestionResult(
        variant=variant.name,
        question=question,
        answer=answer.answer or "",
        tools_used=list(answer.tools_used),
        steps=len(answer.steps),
        citations_found=found,
        citations_missing=missing,
        answer_chars=len(answer.answer or ""),
        latency_ms=(time.perf_counter() - started) * 1000,
    )


def run_eval(
    questions: list[str],
    variants: list[Variant],
    ask_fn: AskFn,
    progress: Callable[[Variant, str], None] | None = None,
) -> list[QuestionResult]:
    results: list[QuestionResult] = []
    for variant in variants:
        for question in questions:
            if progress:
                progress(variant, question)
            started = time.perf_counter()
            try:
                answer = ask_fn(question, variant)
            except Exception as exc:  # noqa: BLE001 - keep the run going
                results.append(
                    QuestionResult(
                        variant=variant.name,
                        question=question,
                        answer="",
                        tools_used=[],
                        steps=0,
                        citations_found=0,
                        citations_missing=0,
                        answer_chars=0,
                        latency_ms=(time.perf_counter() - started) * 1000,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                continue
            results.append(_record(variant, question, answer, started))
    return results


def summarize(results: list[QuestionResult]) -> dict[str, dict]:
    by_variant: dict[str, list[QuestionResult]] = {}
    for r in results:
        by_variant.setdefault(r.variant, []).append(r)

    summary: dict[str, dict] = {}
    for name, rows in by_variant.items():
        n = len(rows)
        found = sum(r.citations_found for r in rows)
        missing = sum(r.citations_missing for r in rows)
        denom = found + missing
        tools: Counter = Counter()
        for r in rows:
            tools.update(r.tools_used)
        summary[name] = {
            "questions": n,
            "errors": sum(1 for r in rows if r.error),
            "avg_steps": round(sum(r.steps for r in rows) / n, 2) if n else 0.0,
            "avg_tools": round(sum(len(r.tools_used) for r in rows) / n, 2) if n else 0.0,
            "tool_counts": dict(tools.most_common()),
            "citations_found": found,
            "citations_missing": missing,
            "citation_found_rate": round(found / denom, 3) if denom else None,
            "avg_answer_chars": round(sum(r.answer_chars for r in rows) / n) if n else 0,
            "avg_latency_ms": round(sum(r.latency_ms for r in rows) / n) if n else 0,
        }
    return summary
