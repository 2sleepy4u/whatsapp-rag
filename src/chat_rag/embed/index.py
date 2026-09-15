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

ProgressFn = Callable[[int, int, dict], None]


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
    # Number of inputs actually sent to the embedder after deduplication, and
    # where the wall time went (Ollama's embedding path has a large per-input
    # overhead, so cutting duplicate inputs matters a lot more than batch size).
    embed_texts: int = 0
    embed_calls: int = 0
    embed_seconds: float = 0.0
    flush_seconds: float = 0.0


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


MAX_UPSERT = 1000


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
    for start in range(0, len(ids), MAX_UPSERT):
        stop = start + MAX_UPSERT
        collection.upsert(
            ids=ids[start:stop],
            embeddings=embeddings[start:stop],
            documents=texts[start:stop],
            metadatas=metadatas[start:stop],
        )
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
    """Embed ``items`` while embedding identical texts only once.

    Ollama's embedding path costs roughly 20-25 ms *per input* regardless of
    text length (and independently of batching or concurrency), so the single
    most effective optimization here is to call it once for each distinct text.
    Chat messages repeat heavily ("ok", greetings, emojis...), so this can cut
    the number of embedding calls by an order of magnitude.
    """
    now = int(time.time())
    total = len(items)
    batch_size = max(1, batch_size)

    by_text: dict[str, list[int]] = {}
    for idx, (_mid, text, _meta) in enumerate(items):
        by_text.setdefault(text, []).append(idx)
    unique_texts = list(by_text.keys())

    buffer: list[tuple[str, str, list[float], dict]] = []
    flush_size = max(batch_size, 512)
    done = 0

    def report() -> None:
        if progress:
            progress(
                done,
                total,
                {
                    "embedded": stats.embed_texts,
                    "embed_total": len(unique_texts),
                    "embed_seconds": stats.embed_seconds,
                    "unit": "emb",
                },
            )

    def flush_buffer() -> None:
        nonlocal buffer
        while buffer:
            part, buffer = buffer[:flush_size], buffer[flush_size:]
            started = time.time()
            _flush(
                conn,
                collection,
                embedder,
                kind,
                [b[0] for b in part],
                [b[1] for b in part],
                [b[2] for b in part],
                [b[3] for b in part],
                now,
            )
            stats.flush_seconds += time.time() - started

    for start in range(0, len(unique_texts), batch_size):
        chunk = unique_texts[start : start + batch_size]
        started = time.time()
        embeddings = embedder.embed(chunk)
        stats.embed_seconds += time.time() - started
        stats.embed_calls += 1
        stats.embed_texts += len(chunk)
        if embeddings and not stats.dim:
            stats.dim = len(embeddings[0])
        for text, vec in zip(chunk, embeddings):
            for idx in by_text[text]:
                buffer.append((items[idx][0], text, vec, items[idx][2]))
                done += 1
        if len(buffer) >= flush_size:
            flush_buffer()
            report()

    flush_buffer()
    stats.embedded = done
    stats.batches = stats.embed_calls
    report()


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
    message_ids: list[str]


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
                    message_ids=[m["id"] for m in chunk],
                )
            )
    return windows


def _fetch_embeddings(collection, ids: list[str], chunk: int = 5000) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for start in range(0, len(ids), chunk):
        res = collection.get(ids=ids[start : start + chunk], include=["embeddings"])
        for mid, vec in zip(res.get("ids", []), res.get("embeddings", [])):
            if vec is not None:
                out[mid] = [float(x) for x in vec]
    return out


def _mean_pool_windows(
    conn,
    collection,
    embedder: EmbeddingClient,
    pending: list[tuple[str, str, dict]],
    windows_by_id: dict[str, _Window],
    stats: IndexStats,
    progress: ProgressFn | None,
) -> None:
    """Window vectors = normalized mean of their member message embeddings.

    This avoids a model call per window entirely. It is a good fit because
    windows exist for *clustering topics* (phase 5), not for precise retrieval,
    and keeps vectors in the same space as the message vectors.
    """
    started = time.time()
    needed: list[str] = []
    seen: set[str] = set()
    for wid, _text, _meta in pending:
        for mid in windows_by_id[wid].message_ids:
            if mid not in seen:
                seen.add(mid)
                needed.append(mid)
    vectors = _fetch_embeddings(collection, needed)
    stats.embed_seconds += time.time() - started

    now = int(time.time())
    ids: list[str] = []
    texts: list[str] = []
    embeddings: list[list[float]] = []
    metadatas: list[dict] = []

    def report(done: int) -> None:
        if progress:
            progress(
                done,
                len(pending),
                {
                    "embedded": done,
                    "embed_total": len(pending),
                    "embed_seconds": stats.embed_seconds,
                    "unit": "pool",
                },
            )

    for i, (wid, text, meta) in enumerate(pending, start=1):
        members = [vectors[mid] for mid in windows_by_id[wid].message_ids if mid in vectors]
        if not members:
            continue
        dim = len(members[0])
        mean = [sum(values for values in group) for group in zip(*members)]
        norm = sum(v * v for v in mean) ** 0.5
        if norm:
            mean = [v / norm for v in mean]
        if not stats.dim:
            stats.dim = dim
        ids.append(wid)
        texts.append(text)
        embeddings.append(mean)
        metadatas.append(meta)
        if len(ids) >= 512:
            _flush(conn, collection, embedder, "window", ids, texts, embeddings, metadatas, now)
            stats.embedded += len(ids)
            ids, texts, embeddings, metadatas = [], [], [], []
            report(i)
    if ids:
        _flush(conn, collection, embedder, "window", ids, texts, embeddings, metadatas, now)
        stats.embedded += len(ids)
    stats.batches = max(1, stats.embedded // 512 + (1 if stats.embedded % 512 else 0))
    report(len(pending))


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
    window_mode: str = "model",
) -> IndexStats:
    if window_mode not in {"model", "mean"}:
        raise ValueError(f"window_mode must be 'model' or 'mean', got {window_mode!r}")
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
        if window_mode == "mean":
            windows_by_id = {w.id: w for w in windows}
            _mean_pool_windows(conn, collection, embedder, pending, windows_by_id, stats, progress)
        else:
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
