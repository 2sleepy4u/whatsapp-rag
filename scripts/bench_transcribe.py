#!/usr/bin/env python
"""Benchmark whisper.cpp (GPU/Vulkan) vs faster-whisper (CPU) on real voice notes.

Examples:

    uv run python scripts/bench_transcribe.py --dir data/exports --limit 5
    uv run python scripts/bench_transcribe.py --dir data/exports \\
        --cpp-bin /path/to/whisper-cli --cpp-model /path/to/ggml-medium.bin

It reports audio-seconds per wall-second (higher is faster). Use it to decide
which engine to run on your hardware; the faster one wins for bulk work.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

AUDIO_EXT = {".opus", ".ogg", ".m4a", ".aac", ".amr", ".wav", ".mp3", ".mp4"}


def collect_files(args) -> list[Path]:
    files: list[Path] = []
    if args.dir:
        files.extend(p for p in Path(args.dir).rglob("*") if p.suffix.lower() in AUDIO_EXT)
    for f in args.file or []:
        files.append(Path(f))
    return files[: args.limit] if args.limit else files


def bench(engine, files: list[Path], label: str) -> None:
    total_audio = 0.0
    total_wall = 0.0
    failures = 0
    for path in files:
        try:
            t0 = time.perf_counter()
            result = engine.transcribe(path)
            wall = time.perf_counter() - t0
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  ! {path.name}: {type(exc).__name__}: {exc}")
            continue
        duration = result.duration or 0.0
        total_audio += duration
        total_wall += wall
        print(f"  {path.name}: {wall:6.1f}s wall, {duration:6.1f}s audio, "
              f"{(duration / wall) if wall else 0:5.2f}x realtime")
    speed = total_audio / total_wall if total_wall else 0.0
    print(f"[{label}] {len(files) - failures}/{len(files)} ok · "
          f"{total_wall:.1f}s wall · {total_audio:.1f}s audio · {speed:.2f}x realtime\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", help="Folder to search for audio files")
    parser.add_argument("--file", action="append", help="Specific audio file (repeatable)")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--language", default="it")
    parser.add_argument("--whisper-model", default="small", help="faster-whisper model")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--cpp-bin", help="whisper.cpp binary")
    parser.add_argument("--cpp-model", help="whisper.cpp ggml model")
    parser.add_argument("--engine", choices=["auto", "faster-whisper", "whisper.cpp"], default="auto")
    args = parser.parse_args()

    files = collect_files(args)
    if not files:
        print("No audio files found. Use --dir or --file.", file=sys.stderr)
        return 1
    print(f"Benchmarking {len(files)} file(s)\n")

    if args.engine in {"auto", "faster-whisper"}:
        try:
            from chat_rag.transcribe.faster_whisper_engine import FasterWhisperEngine

            engine = FasterWhisperEngine(model=args.whisper_model, language=args.language,
                                         compute_type=args.compute_type)
            bench(engine, files, f"faster-whisper/{args.whisper_model}")
        except Exception as exc:  # noqa: BLE001
            print(f"[faster-whisper] unavailable: {exc}\n")

    if args.engine in {"auto", "whisper.cpp"} and args.cpp_bin and args.cpp_model:
        from chat_rag.transcribe.whisper_cpp_engine import WhisperCppEngine

        engine = WhisperCppEngine(args.cpp_bin, args.cpp_model, language=args.language)
        bench(engine, files, "whisper.cpp")
    elif args.engine == "whisper.cpp":
        print("whisper.cpp needs --cpp-bin and --cpp-model", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
