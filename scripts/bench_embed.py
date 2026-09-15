#!/usr/bin/env python3
"""Micro-benchmark for Ollama embedding throughput.

Reads real message texts from the ingested SQLite DB (so text lengths match
production), then measures texts/sec and (when Ollama reports it) tokens/sec
across batch sizes and models. Also supports a CPU-only run to compare against
the Vulkan GPU path.

Usage::

    uv run python scripts/bench_embed.py --model bge-m3 --length short
    uv run python scripts/bench_embed.py --model embeddinggemma --length window
    uv run python scripts/bench_embed.py --model bge-m3 --cpu
"""

from __future__ import annotations

import argparse
import sqlite3
import statistics
import time
from pathlib import Path

import ollama

DB = Path("data/chat.db")


def load_texts(kind: str, limit: int = 512) -> list[str]:
    conn = sqlite3.connect(DB)
    rows = conn.execute(
        "SELECT text FROM messages WHERE msg_type IN ('text','edited') ORDER BY ts LIMIT ?",
        (limit * 8,),
    ).fetchall()
    conn.close()
    texts = [r[0] for r in rows]
    if kind == "short":
        return texts[:limit]
    windows = ["\n".join(texts[i : i + 6]) for i in range(0, len(texts) - 6, 3)]
    return windows[:limit]


def bench(model: str, texts: list[str], sizes: list[int], calls: int, cpu: bool) -> None:
    host = "http://localhost:11434"
    client = ollama.Client(host=host)
    options = {"num_gpu": 0} if cpu else None

    print(f"\nmodel={model} length={len(texts[0])} chars  n_texts={len(texts)}  cpu={cpu}")
    print(f"{'batch':>6} {'calls':>6} {'texts':>7} {'sec':>8} {'texts/s':>9} {'tokens/s':>9} {'ms/call':>8}")

    # warm-up: forces model load so it doesn't pollute the first measurement
    client.embed(model=model, input=texts[:8], options=options, keep_alive="30m")

    for size in sizes:
        if size > len(texts):
            continue
        samples: list[float] = []
        tokens_seen = 0
        n_texts = 0
        for _ in range(calls):
            batch = texts[:size]
            t0 = time.perf_counter()
            resp = client.embed(model=model, input=batch, options=options, keep_alive="30m")
            samples.append(time.perf_counter() - t0)
            n_texts += len(batch)
            tokens_seen += getattr(resp, "prompt_eval_count", 0) or 0
        total = sum(samples)
        tps = n_texts / total if total else 0
        tokps = tokens_seen / total if tokens_seen else float("nan")
        print(
            f"{size:>6} {calls:>6} {n_texts:>7} {total:>8.3f} {tps:>9.1f} "
            f"{tokps:>9.1f} {statistics.mean(samples) * 1000:>8.1f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="bge-m3")
    ap.add_argument("--length", choices=["short", "window"], default="short")
    ap.add_argument("--sizes", default="1,8,16,32,64,128,256")
    ap.add_argument("--calls", type=int, default=5)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    texts = load_texts(args.length)
    sizes = [int(s) for s in args.sizes.split(",")]
    bench(args.model, texts, sizes, args.calls, args.cpu)


if __name__ == "__main__":
    main()
