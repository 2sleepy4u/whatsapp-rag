"""Voice-note transcription pipeline.

Discovers ``voice`` messages with a media reference, locates the audio file
next to the original export (or in an extra search root), transcribes it with
the configured engine, stores the transcript, and (optionally) replaces the
placeholder ``messages.text`` so the note becomes searchable and citable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote

from .base import TranscriptionEngine


@dataclass
class TranscribeStats:
    engine: str = ""
    model: str = ""
    total: int = 0
    transcribed: int = 0
    skipped: int = 0
    missing: int = 0
    failed: int = 0
    dry_run: bool = False
    errors: list[tuple[str, str]] = field(default_factory=list)
    elapsed: float = 0.0


def media_search_roots(conn, extra: list[Path] | None = None) -> list[Path]:
    roots: list[Path] = []
    for row in conn.execute("SELECT DISTINCT source_file FROM messages WHERE source_file IS NOT NULL"):
        parent = Path(row["source_file"]).expanduser()
        roots.append(parent.parent if parent.is_file() else parent)
    roots.extend(Path(p).expanduser() for p in (extra or []))
    seen: dict[str, Path] = {}
    for r in roots:
        if r.exists():
            seen.setdefault(str(r.resolve()), r)
    return list(seen.values())


def build_media_index(roots: list[Path]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for root in roots:
        for path in root.rglob("*"):
            if path.is_file():
                index.setdefault(path.name, path)
    return index


def voice_messages(conn, chat_id: str | None = None):
    clauses = ["msg_type = 'voice'"]
    params: list = []
    if chat_id:
        clauses.append("chat_id = ?")
        params.append(chat_id)
    return conn.execute(
        f"SELECT id, chat_id, media_ref, text, ts_local FROM messages WHERE {' AND '.join(clauses)} "
        "ORDER BY ts, rowid",
        params,
    ).fetchall()


def _already(conn, message_id: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM transcripts WHERE message_id = ?", (message_id,)
    ).fetchone() is not None


def _store(conn, message_id: str, media_file: str, engine: TranscriptionEngine,
           language: str | None, duration: float | None, text: str, apply_to_messages: bool) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO transcripts "
        "(message_id, media_file, engine, model, language, duration_sec, text, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (message_id, media_file, engine.name, engine.model, language, duration, text, int(time.time())),
    )
    if apply_to_messages:
        conn.execute("UPDATE messages SET text = ? WHERE id = ?", (text, message_id))
    conn.commit()


def transcribe_pending(
    conn,
    engine: TranscriptionEngine | None,
    roots: list[Path],
    chat_id: str | None = None,
    limit: int | None = None,
    apply_to_messages: bool = True,
    overwrite: bool = False,
    dry_run: bool = False,
    on_progress=None,
) -> TranscribeStats:
    rows = voice_messages(conn, chat_id)
    if limit:
        rows = rows[:limit]

    stats = TranscribeStats(dry_run=dry_run)
    if engine is not None:
        stats.engine = engine.name
        stats.model = engine.model
    stats.total = len(rows)
    started = time.time()

    index = build_media_index(roots)

    for i, row in enumerate(rows, start=1):
        if on_progress:
            on_progress(i, len(rows), row["media_ref"] or "(nessun file)")
        if not overwrite and _already(conn, row["id"]):
            stats.skipped += 1
            continue

        media_ref = row["media_ref"]
        path = None
        if media_ref:
            path = index.get(media_ref) or index.get(unquote(media_ref))
        if path is None:
            stats.missing += 1
            continue

        if dry_run:
            stats.transcribed += 1
            continue
        assert engine is not None
        try:
            result = engine.transcribe(path)
        except Exception as exc:  # noqa: BLE001 - keep going on bad files
            stats.failed += 1
            stats.errors.append((row["id"], f"{type(exc).__name__}: {exc}"))
            continue

        _store(conn, row["id"], str(path.name), engine, result.language, result.duration,
               result.text, apply_to_messages)
        stats.transcribed += 1

    stats.elapsed = time.time() - started
    return stats
