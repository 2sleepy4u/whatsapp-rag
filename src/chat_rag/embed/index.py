"""Build and refresh the Chroma vector store from the SQLite messages table.

Two kinds of vectors are produced:

* ``message`` — one vector per message (best for precise retrieval/citations);
* ``window``  — sliding windows of consecutive messages (better topics for
  clustering, since single chat messages are very short).

Indexing is incremental: an ``embedding_index`` row records what has already
been embedded for a given ``(collection, kind, model)`` and the hash of the
text. Re-running only embeds new or changed messages.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import chromadb
from chromadb.config import Settings as ChromaSettings

from ..embed.base import EmbeddingClient

INDEXABLE_TYPES = ("text", "edited", "voice")
ENTITY_TYPES = ("text", "edited", "voice", "media", "image", "video", "document", "sticker", "gif", "location")

ProgressFn = Callable[[int, int], None]


def chroma_client(path: Path) -> chromadb.ClientAPI:
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(path),
        settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
    )


def ensure_collection(client: chromadb.ClientAPI, name: str):
    return client.get_or_create_collection(
        name=name, metadata={"hnsw:space": "cosine"}
    )


def _text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()


def _sender_name(row) -> str:
    return row["sender_name"] or (row["sender_id"].split("::", 1)[1] if row["sender_id"] and "::" in row["sender_id"] else "")


def _message_text(row) -> str:
    name = _sender_name(row)
    return f"{name}: {row['text']}" if name else row["text"]


@dataclass
class IndexStats:
    kind: str
    model: str
    total: int = 0
    embedded: int = 0
    skipped: int = 0
    pending: int = 0
    dim: int = 0
    batches: int = 0
    elapsed: float = 0.0
    dry_run: bool = False


def _fetch_messages(conn, chat_id: str | None, types: tuple[str, ...]):
    where = ["m.msg_type IN (%s)" % ",".join("?" * len(types))]
    params: list = list(types)
    if chat_id:
        where.append("m.chat_id = ?")
        params.append(chat_id)
    sql = f"""
        SELECT m.id, m.chat_id, m.sender_id, m.ts, m.local_date, m.msg_type, m.text,
               s.name AS sender_name
        FROM messages m
        LEFT JOIN senders s ON s.sender_id = m.sender_id
        WHERE {' AND '.join(where)}
        ORDER BY m.chat_id, m.ts, m.rowid
    """
    return conn.execute(sql, params).fetchall()


def _existing_hashes(conn, collection: str, model: str, kind: str) -> dict[str, str]:
    rows = conn.execute(
        "SELECT id, text_hash FROM embedding_index WHERE collection = ? AND model = ? AND kind = ?",
        (collection, model, kind),
    ).fetchall()
    return {r["id"]: r["text_hash"] for r in rows}


def _flush(
    conn,
    collection,
    embedder: EmbeddingClient,
    kind: str,
    ids: list[str],
    texts: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict],
    now: int,
) -> None:
    collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    conn.executemany(
        "INSERT OR REPLACE INTO embedding_index (id, collection, kind, model, text_hash, ts, indexed_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (i, collection.name, kind, embedder.model, _text_hash(t), md.get("ts"), now)
            for i, t, md in zip(ids, texts, metadatas)
        ],
    )
    conn.commit()


def _run_batches(
    conn,
    collection,
    embedder: EmbeddingClient,
    kind: str,
    items: list[tuple[str, str, dict]],
    batch_size: int,
    stats: IndexStats,
    progress: ProgressFn | None,
) -> None:
    now = int(time.time())
    done = 0
    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        ids = [c[0] for c in chunk]
        texts = [c[1] for c in chunk]
        metadatas = [c[2] for c in chunk]
        embeddings = embedder.embed(texts)
        if embeddings and not stats.dim:
            stats.dim = len(embeddings[0])
        _flush(conn, collection, embedder, kind, ids, texts, embeddings, metadatas, now)
        stats.embedded += len(ids)
        stats.batches += 1
        done += len(ids)
        if progress:
            progress(done, len(items))


def _prepare(
    conn,
    collection,
    embedder: EmbeddingClient,
    kind: str,
    candidates: list[tuple[str, str, dict]],
    recreate: bool,
) -> list[tuple[str, str, dict]]:
    existing = {} if recreate else _existing_hashes(conn, collection.name, embedder.model, kind)
    pending: list[tuple[str, str, dict]] = []
    for cid, text, meta in candidates:
        prev = existing.get(cid)
        if prev is not None and prev == _text_hash(text):
            continue
        pending.append((cid, text, meta))
    return pending


def index_messages(
    conn,
    collection,
    embedder: EmbeddingClient,
    chat_id: str | None = None,
    types: tuple[str, ...] = INDEXABLE_TYPES,
    batch_size: int = 64,
    recreate: bool = False,
    limit: int | None = None,
    progress: ProgressFn | None = None,
    dry_run: bool = False,
) -> IndexStats:
    stats = IndexStats(kind="message", model=embedder.model, dry_run=dry_run)
    started = time.time()
    rows = _fetch_messages(conn, chat_id, types)
    stats.total = len(rows)
    candidates = [
        (
            row["id"],
            _message_text(row),
            {
                "chat_id": row["chat_id"],
                "sender_id": row["sender_id"] or "",
                "sender_name": _sender_name(row),
                "ts": int(row["ts"]),
                "local_date": row["local_date"],
                "msg_type": row["msg_type"],
                "kind": "message",
            },
        )
        for row in rows
    ]
    if limit:
        candidates = candidates[:limit]
    pending = _prepare(conn, collection, embedder, "message", candidates, recreate)
    stats.skipped = stats.total - len(pending)
    stats.pending = len(pending)
    if not dry_run:
        _run_batches(conn, collection, embedder, "message", pending, batch_size, stats, progress)
    stats.elapsed = time.time() - started
    return stats


@dataclass
class _Window:
    id: str
    text: str
    meta: dict


def build_windows(rows: Iterable, size: int, stride: int) -> list[_Window]:
    by_chat: dict[str, list] = {}
    for row in rows:
        by_chat.setdefault(row["chat_id"], []).append(row)

    windows: list[_Window] = []
    for chat, msgs in by_chat.items():
        if not msgs:
            continue
        n = len(msgs)
        if n <= size:
            starts = [0]
        else:
            starts = list(range(0, n - size + 1, stride))
            if starts[-1] != n - size:
                starts.append(n - size)

        seen: set[str] = set()
        for s in starts:
            chunk = msgs[s : s + size]
            first, last = chunk[0], chunk[-1]
            wid = f"{chat}:w:{first['id']}_{last['id']}"
            if wid in seen:
                continue
            seen.add(wid)
            text = "\n".join(_message_text(m) for m in chunk)
            windows.append(
                _Window(
                    id=wid,
                    text=text,
                    meta={
                        "chat_id": chat,
                        "kind": "window",
                        "start_id": first["id"],
                        "end_id": last["id"],
                        "start_ts": int(first["ts"]),
                        "end_ts": int(last["ts"]),
                        "start_date": first["local_date"],
                        "end_date": last["local_date"],
                        "messages": len(chunk),
                    },
                )
            )
    return windows


def index_windows(
    conn,
    collection,
    embedder: EmbeddingClient,
    chat_id: str | None = None,
    size: int = 6,
    stride: int = 3,
    batch_size: int = 32,
    recreate: bool = False,
    limit: int | None = None,
    progress: ProgressFn | None = None,
    dry_run: bool = False,
) -> IndexStats:
    stats = IndexStats(kind="window", model=embedder.model, dry_run=dry_run)
    started = time.time()
    rows = _fetch_messages(conn, chat_id, INDEXABLE_TYPES)
    windows = build_windows(rows, size, stride)
    stats.total = len(windows)
    candidates = [(w.id, w.text, w.meta) for w in windows]
    if limit:
        candidates = candidates[:limit]
    pending = _prepare(conn, collection, embedder, "window", candidates, recreate)
    stats.skipped = stats.total - len(pending)
    stats.pending = len(pending)
    if not dry_run:
        _run_batches(conn, collection, embedder, "window", pending, batch_size, stats, progress)
    stats.elapsed = time.time() - started
    return stats


def reset_index(conn, client: chromadb.ClientAPI, collection_name: str, model: str) -> None:
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    conn.execute("DELETE FROM embedding_index WHERE collection = ? AND model = ?", (collection_name, model))
    conn.commit()
