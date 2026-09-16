#!/usr/bin/env python3
"""A/B evaluation for the RAG agent on a set of questions.

Runs every question under one or more variants (prompt file and/or thinking mode)
and prints an aggregate table plus per-question results. Requires a running
Ollama with the configured LLM/embedding models; use the hermetic pytest suite
for logic-only checks.

Usage::

    uv run python scripts/eval_rag.py --questions questions.json \
        --prompt prompts/strict.txt --think auto,on --out report.json

Questions file format: a JSON list of strings, or of ``{"question": "..."}``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from chat_rag.config import load_settings
from chat_rag.rag.agent import OllamaLLM
from chat_rag.rag.eval import Variant, load_questions, run_eval, summarize
from chat_rag.rag.service import answer_question, build_context

_THINK = {"auto": None, "off": False, "on": True}


def _suffix(think: bool | None) -> str:
    return {None: "-auto", False: "-think-off", True: "-think-on"}[think]


def build_variants(prompt_path: str | None, thinks: list[bool | None]) -> list[Variant]:
    base_prompt = Path(prompt_path).read_text(encoding="utf-8") if prompt_path else None
    variants: list[Variant] = []
    for think in thinks:
        variants.append(Variant(name=f"base{_suffix(think)}", system_override=None, think=think))
        if base_prompt is not None:
            variants.append(
                Variant(name=f"prompt{_suffix(think)}", system_override=base_prompt, think=think)
            )
    return variants


def make_ask(settings, ctx, max_steps: int):
    cache: dict[tuple[str, bool | None], OllamaLLM] = {}

    def ask(question: str, variant: Variant):
        key = (settings.llm_model, variant.think)
        llm = cache.get(key)
        if llm is None:
            llm = OllamaLLM(
                settings.llm_model,
                settings.ollama_host,
                num_predict=settings.llm_num_predict,
                think=variant.think,
            )
            cache[key] = llm
        return answer_question(
            settings,
            question,
            llm=llm,
            ctx=ctx,
            max_steps=max_steps,
            system_override=variant.system_override,
        )

    return ask


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--questions", required=True, help="JSON file with the questions")
    ap.add_argument("--prompt", help="Alternative system prompt file to A/B")
    ap.add_argument("--think", default="auto,on", help="Comma list of auto|off|on")
    ap.add_argument("--model", help="Override CHAT_RAG_LLM_MODEL for this run")
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--out", help="Write raw per-question results as JSON")
    args = ap.parse_args()

    settings = load_settings()
    if args.model:
        settings = replace(settings, llm_model=args.model)

    questions = load_questions(args.questions)
    thinks = [_THINK[t.strip()] for t in args.think.split(",") if t.strip()]
    variants = build_variants(args.prompt, thinks)

    settings.ensure_dirs()
    ctx = build_context(settings)
    ask = make_ask(settings, ctx, args.max_steps)

    print(f"model={settings.llm_model}  questions={len(questions)}  variants={len(variants)}")
    results = run_eval(
        questions,
        variants,
        ask,
        progress=lambda v, q: print(f"  [{v.name}] {q[:70]}", flush=True),
    )
    ctx.conn.close()

    report = summarize(results)
    print("\n=== summary ===")
    header = f"{'variant':<20} {'n':>3} {'err':>4} {'steps':>6} {'cites':>7} {'found%':>7} {'chars':>7} {'ms':>7}"
    print(header)
    for name, s in report.items():
        rate = "n/a" if s["citation_found_rate"] is None else f"{s['citation_found_rate'] * 100:.0f}%"
        print(
            f"{name:<20} {s['questions']:>3} {s['errors']:>4} {s['avg_steps']:>6} "
            f"{s['citations_found']:>7} {rate:>7} {s['avg_answer_chars']:>7} {s['avg_latency_ms']:>7}"
        )
        if s["tool_counts"]:
            print(f"    tools: {s['tool_counts']}")

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {"summary": report, "results": [r.to_dict() for r in results]},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
