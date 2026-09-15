from __future__ import annotations

import hashlib
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

from .parser_android import ParseStats, ParsedMessage, chat_id_from_filename, parse_file

INSERT_SQL = """
INSERT OR IGNORE INTO messages
    (id, chat_id, sender_id, ts, ts_local, local_date, local_time, text,
     raw_line, msg_type, reply_to_id, media_ref, source_file, line_no, ingested_at)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

BATCH = 2000


@dataclass
class IngestResult:
    source_file: str
    chat_id: str
    file_hash: str
    total_lines: int = 0
    parsed: int = 0
    inserted: int = 0
    skipped: int = 0
    continuations: int = 0
    by_type: dict[str, int] = field(default_factory=dict)
    senders: dict[str, int] = field(default_factory=dict)
    first_ts: int | None = None
    last_ts: int | None = None
    already_ingested: bool = False

    def summary(self) -> str:
        kinds = ", ".join(f"{k}={v}" for k, v in sorted(self.by_type.items()))
        return (
            f"chat={self.chat_id} file={self.source_file}\n"
            f"  lines={self.total_lines} parsed={self.parsed} inserted={self.inserted} "
            f"skipped={self.skipped} continuations={self.continuations}\n"
            f"  types: {kinds or '-'}\n"
            f"  senders: {len(self.senders)}"
        )


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sender_id(chat_id: str, name: str) -> str:
    return f"{chat_id}::{name}"


def _base_id(chat_id: str, ts: int, sender: str | None, text: str, raw: str) -> str:
    payload = f"{chat_id}|{ts}|{sender or ''}|{text}|{raw}".encode("utf-8", "replace")
    return hashlib.sha1(payload).hexdigest()[:16]


def _already_ingested(conn: sqlite3.Connection, source_file: str, file_hash: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM ingestion_runs WHERE source_file = ? AND file_hash = ? "
        "AND status = 'done' LIMIT 1",
        (source_file, file_hash),
    ).fetchone()
    return row is not None


def ingest_file(
    conn: sqlite3.Connection,
    path: Path,
    chat_id: str | None = None,
    tz: str = "Europe/Rome",
    force: bool = False,
) -> IngestResult:
    chat_id = chat_id or chat_id_from_filename(path)
    file_hash = file_sha256(path)
    source_file = str(path)

    if not force and _already_ingested(conn, source_file, file_hash):
        return IngestResult(
            source_file=source_file,
            chat_id=chat_id,
            file_hash=file_hash,
            already_ingested=True,
        )

    now = int(time.time())
    run_id = conn.execute(
        "INSERT INTO ingestion_runs (source_file, file_hash, chat_id, started_at) "
        "VALUES (?, ?, ?, ?)",
        (source_file, file_hash, chat_id, now),
    ).lastrowid
    conn.execute(
        "INSERT INTO chats (chat_id, name, source_file) VALUES (?, ?, ?) "
        "ON CONFLICT(chat_id) DO UPDATE SET source_file = excluded.source_file",
        (chat_id, chat_id, source_file),
    )

    result = IngestResult(source_file=source_file, chat_id=chat_id, file_hash=file_hash)
    occ: dict[str, int] = {}
    sender_seen: set[str] = set()
    batch: list[tuple] = []
    final_stats = ParseStats()
    message_total = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]

    for msg, stats in parse_file(path, tz_name=tz):
        final_stats = stats
        result.parsed += 1
        result.by_type[msg.msg_type] = result.by_type.get(msg.msg_type, 0) + 1

        base = _base_id(chat_id, msg.ts, msg.sender, msg.text, msg.raw_line)
        n = occ.get(base, 0)
        occ[base] = n + 1
        msg_id = base if n == 0 else f"{base}-{n}"

        sender_id = None
        if msg.sender is not None:
            sender_id = _sender_id(chat_id, msg.sender)
            result.senders[msg.sender] = result.senders.get(msg.sender, 0) + 1
            if msg.sender not in sender_seen:
                sender_seen.add(msg.sender)
                conn.execute(
                    "INSERT INTO senders (sender_id, chat_id, name) VALUES (?, ?, ?) "
                    "ON CONFLICT(sender_id) DO NOTHING",
                    (sender_id, chat_id, msg.sender),
                )

        batch.append(
            (
                msg_id,
                chat_id,
                sender_id,
                msg.ts,
                msg.ts_local,
                msg.local_date,
                msg.local_time,
                msg.text,
                msg.raw_line,
                msg.msg_type,
                None,
                msg.media_ref,
                source_file,
                msg.line_no,
                now,
            )
        )
        if msg.ts:
            if result.first_ts is None or msg.ts < result.first_ts:
                result.first_ts = msg.ts
            if result.last_ts is None or msg.ts > result.last_ts:
                result.last_ts = msg.ts

        if len(batch) >= BATCH:
            inserted, message_total = _flush(conn, batch, message_total)
            result.inserted += inserted

    if batch:
        inserted, message_total = _flush(conn, batch, message_total)
        result.inserted += inserted

    result.total_lines = final_stats.lines
    result.continuations = final_stats.continuations
    result.skipped = result.parsed - result.inserted

    _finalize(conn, chat_id, result, run_id=run_id)
    return result


def _flush(conn: sqlite3.Connection, batch: list[tuple], previous_total: int) -> tuple[int, int]:
    """Insert a batch and return (inserted, new_total).

    ``INSERT OR IGNORE`` makes the exact inserted count necessary; we take a
    single COUNT(*) instead of two to keep the O(n) scans down on large imports.
    """
    conn.executemany(INSERT_SQL, batch)
    conn.commit()
    total = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    batch.clear()
    return total - previous_total, total


def _finalize(conn: sqlite3.Connection, chat_id: str, result: IngestResult, run_id: int) -> None:
    conn.execute(
        """
        UPDATE chats SET
            is_group = (
                (SELECT COUNT(DISTINCT sender_id) FROM messages
                 WHERE chat_id = ? AND sender_id IS NOT NULL) > 2
                OR EXISTS (
                    SELECT 1 FROM messages
                    WHERE chat_id = ? AND msg_type = 'system'
                      AND (lower(text) LIKE '%gruppo%' OR lower(text) LIKE '%group%')
                )
            ),
            first_ts = (SELECT MIN(ts) FROM messages WHERE chat_id = ?),
            last_ts  = (SELECT MAX(ts) FROM messages WHERE chat_id = ?),
            message_count = (SELECT COUNT(*) FROM messages WHERE chat_id = ?)
        WHERE chat_id = ?
        """,
        (chat_id, chat_id, chat_id, chat_id, chat_id, chat_id),
    )
    conn.execute(
        """
        UPDATE senders SET
            first_ts = (SELECT MIN(ts) FROM messages WHERE sender_id = senders.sender_id),
            last_ts  = (SELECT MAX(ts) FROM messages WHERE sender_id = senders.sender_id),
            message_count = (SELECT COUNT(*) FROM messages WHERE sender_id = senders.sender_id)
        WHERE chat_id = ?
        """,
        (chat_id,),
    )
    conn.execute(
        "UPDATE ingestion_runs SET finished_at = ?, total_lines = ?, parsed = ?, "
        "inserted = ?, skipped = ?, continuations = ?, status = 'done' WHERE id = ?",
        (
            int(time.time()),
            result.total_lines,
            result.parsed,
            result.inserted,
            result.skipped,
            result.continuations,
            run_id,
        ),
    )
    conn.commit()
    conn.execute("INSERT INTO messages_fts(messages_fts) VALUES ('optimize')")
    conn.commit()
