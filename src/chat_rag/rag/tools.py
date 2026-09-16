"""Tools the local LLM can call.

Each tool is a plain function over a :class:`ToolContext` (SQLite connection,
Chroma collection, embedder). Tools return JSON-serialisable structures that
always carry a stable ``id`` for every message they surface, so the agent can
cite them and the CLI can expand them.

The module also exposes ``TOOL_SCHEMAS`` (Ollama function-calling schemas) and
``dispatch`` (name -> callable).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Callable

from ..analytics import clustering as clustering_mod
from ..analytics import inside_jokes
from ..db import stats as stats_mod

MAX_TEXT = 400

# --- context ---------------------------------------------------------------


@dataclass
class ToolContext:
    conn: Any
    collection: Any
    embedder: Any
    llm: Any | None = None


# --- helpers ---------------------------------------------------------------


def _sender_name(sender_id: str | None) -> str:
    if not sender_id:
        return ""
    return sender_id.split("::", 1)[1] if "::" in sender_id else sender_id


def _clip(text: str, limit: int = MAX_TEXT) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _message_dict(row) -> dict:
    keys = row.keys()
    return {
        "id": row["id"],
        "date": row["local_date"],
        "time": row["local_time"],
        "sender": _sender_name(row["sender_id"]) if "sender_id" in keys else "",
        "text": _clip(row["text"]),
        "type": row["msg_type"] if "msg_type" in keys else "text",
    }


def _records_to_dump(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    return obj


def _messages_by_ids(conn, ids: list[str]) -> list[dict]:
    """Fetch real messages by id, preserving the order of ``ids``."""
    if not ids:
        return []
    placeholders = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT id, local_date, local_time, sender_id, text, msg_type "
        f"FROM messages WHERE id IN ({placeholders})",
        ids,
    ).fetchall()
    by_id = {r["id"]: _message_dict(r) for r in rows}
    return [by_id[i] for i in ids if i in by_id]


def _build_where(kind: str | None, chat_id: str | None, sender: str | None,
                 date_from: str | None, date_to: str | None) -> dict | None:
    clauses: list[dict] = []
    if kind:
        clauses.append({"kind": kind})
    if chat_id:
        clauses.append({"chat_id": chat_id})
    if sender:
        clauses.append({"sender_name": sender})
    if date_from:
        clauses.append({"local_date": {"$gte": date_from}})
    if date_to:
        clauses.append({"local_date": {"$lte": date_to}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def _window_where(chat_id: str | None, date_from: str | None, date_to: str | None) -> dict:
    clauses: list[dict] = [{"kind": "window"}]
    if chat_id:
        clauses.append({"chat_id": chat_id})
    if date_from:
        clauses.append({"start_date": {"$gte": date_from}})
    if date_to:
        clauses.append({"end_date": {"$lte": date_to}})
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


# --- tool implementations --------------------------------------------------


def semantic_search(
    ctx: ToolContext,
    query: str,
    top: int = 8,
    chat_id: str | None = None,
    sender: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    context: int = 0,
) -> list[dict]:
    where = _build_where("message", chat_id, sender, date_from, date_to)
    qvec = ctx.embedder.embed([query])[0]
    res = ctx.collection.query(
        query_embeddings=[qvec],
        n_results=max(1, min(top, 50)),
        where=where or None,
    )
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    out = []
    for mid, doc, meta, dist in zip(ids, docs, metas, dists):
        out.append(
            {
                "id": mid,
                "date": meta.get("local_date", ""),
                "sender": meta.get("sender_name", ""),
                "text": _clip(doc),
                "score": round(1 - dist, 4) if dist is not None else None,
            }
        )
    return _attach_context(ctx, out, context)


def _attach_context(ctx: ToolContext, rows: list[dict], context: int) -> list[dict]:
    """Nest the real messages surrounding each hit so the model gets the exchange."""
    context = max(0, min(int(context), 10))
    if not context:
        return rows
    for row in rows:
        data = get_context(ctx, row["id"], before=context, after=context)
        row["context"] = [m for m in data.get("messages", []) if m["id"] != row["id"]]
    return rows


def _rrf(rank_lists: list[list[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion over ranked id lists (higher is better)."""
    scores: dict[str, float] = {}
    for ranks in rank_lists:
        for position, mid in enumerate(ranks):
            scores[mid] = scores.get(mid, 0.0) + 1.0 / (k + position + 1)
    return scores


def hybrid_search(
    ctx: ToolContext,
    query: str,
    top: int = 10,
    context: int = 0,
    chat_id: str | None = None,
    sender: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[dict]:
    """Semantic + keyword results fused with RRF (best default for factual lookups)."""
    top = max(1, min(top, 50))
    pool = min(top * 2, 50)
    sem = semantic_search(
        ctx, query, top=pool, chat_id=chat_id, sender=sender,
        date_from=date_from, date_to=date_to,
    )
    kw = keyword_search(ctx, query, top=pool, chat_id=chat_id, date_from=date_from, date_to=date_to)

    by_id: dict[str, dict] = {}
    for rec in sem:
        by_id.setdefault(rec["id"], rec)
    for rec in kw:
        by_id.setdefault(rec["id"], rec)

    sem_rank = {rec["id"]: i + 1 for i, rec in enumerate(sem)}
    kw_rank = {rec["id"]: i + 1 for i, rec in enumerate(kw)}
    scores = _rrf([list(sem_rank), list(kw_rank)])
    ranked = sorted(scores, key=lambda mid: scores[mid], reverse=True)[:top]

    out: list[dict] = []
    for mid in ranked:
        rec = dict(by_id[mid])
        channels = []
        if mid in sem_rank:
            channels.append("semantic")
        if mid in kw_rank:
            channels.append("keyword")
        rec["score"] = round(scores[mid], 6)
        rec["channels"] = channels
        rec["semantic_rank"] = sem_rank.get(mid)
        rec["keyword_rank"] = kw_rank.get(mid)
        out.append(rec)
    return _attach_context(ctx, out, context)


def smart_search(
    ctx: ToolContext,
    query: str,
    top: int = 10,
    context: int = 2,
    expand: int = 3,
    use_hyde: bool = False,
    chat_id: str | None = None,
    sender: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[dict]:
    """Best-recall search: LLM query expansion + HyDE, fused with RRF.

    Falls back to ``hybrid_search`` when no LLM is available in the context.
    """
    filters = dict(
        chat_id=chat_id, sender=sender, date_from=date_from, date_to=date_to
    )
    if ctx.llm is None:
        return hybrid_search(ctx, query, top=top, context=context, **filters)

    from .query import expand_queries, hyde_document

    queries = [query] + expand_queries(ctx.llm, query, n=max(0, min(expand, 5)))
    if use_hyde:
        draft = hyde_document(ctx.llm, query)
        if draft:
            queries.append(draft)

    top = max(1, min(top, 50))
    pool = min(top * 2, 50)
    by_id: dict[str, dict] = {}
    sem_lists: list[list[str]] = []
    kw_lists: list[list[str]] = []
    for q in queries:
        sem = semantic_search(ctx, q, top=pool, **filters)
        sem_lists.append([r["id"] for r in sem])
        for rec in sem:
            by_id.setdefault(rec["id"], rec)
    for q in queries:
        kw = keyword_search(ctx, q, top=pool, chat_id=chat_id, date_from=date_from, date_to=date_to)
        kw_lists.append([r["id"] for r in kw])
        for rec in kw:
            by_id.setdefault(rec["id"], rec)

    scores = _rrf(sem_lists + kw_lists)
    ranked = sorted(scores, key=lambda mid: scores[mid], reverse=True)[:top]
    sem_ids = {mid for lst in sem_lists for mid in lst}
    kw_ids = {mid for lst in kw_lists for mid in lst}
    out: list[dict] = []
    for mid in ranked:
        rec = dict(by_id[mid])
        rec["score"] = round(scores[mid], 6)
        rec["channels"] = [c for c, ids in (("semantic", sem_ids), ("keyword", kw_ids)) if mid in ids]
        out.append(rec)
    return _attach_context(ctx, out, context)


def window_search(
    ctx: ToolContext,
    query: str,
    top: int = 5,
    chat_id: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    max_messages: int = 20,
) -> list[dict]:
    """Search conversation windows, returning the real messages inside each hit."""
    qvec = ctx.embedder.embed([query])[0]
    res = ctx.collection.query(
        query_embeddings=[qvec],
        n_results=max(1, min(top, 25)),
        where=_window_where(chat_id, date_from, date_to),
    )
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    out: list[dict] = []
    for wid, doc, meta, dist in zip(ids, docs, metas, dists):
        raw = str(meta.get("message_ids") or "")
        member_ids = [x for x in raw.split(",") if x]
        if not member_ids and meta.get("start_id"):
            member_ids = [meta["start_id"]]
        messages = _messages_by_ids(ctx.conn, member_ids)[: max(1, min(max_messages, 50))]
        out.append(
            {
                "window_id": wid,
                "chat_id": meta.get("chat_id", ""),
                "start_date": meta.get("start_date", ""),
                "end_date": meta.get("end_date", ""),
                "score": round(1 - dist, 4) if dist is not None else None,
                "text": _clip(doc, 1200),
                "messages": messages,
            }
        )
    return out


def _fts_query(query: str) -> str:
    tokens = [t for t in re.findall(r"\w+", query, re.UNICODE) if t]
    if not tokens:
        return '""'
    return " AND ".join(f'"{t}"' for t in tokens)


def keyword_search(
    ctx: ToolContext,
    query: str,
    top: int = 20,
    chat_id: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[dict]:
    clauses = []
    params: list = [_fts_query(query)]
    if chat_id:
        clauses.append("m.chat_id = ?")
        params.append(chat_id)
    if date_from:
        clauses.append("m.local_date >= ?")
        params.append(date_from)
    if date_to:
        clauses.append("m.local_date <= ?")
        params.append(date_to)
    extra = (" AND " + " AND ".join(clauses)) if clauses else ""
    sql = f"""
        SELECT m.id, m.local_date, m.local_time, m.sender_id, m.text, m.msg_type
        FROM messages_fts f
        JOIN messages m ON m.rowid = f.rowid
        WHERE messages_fts MATCH ?{extra}
        ORDER BY rank
        LIMIT ?
    """
    params.append(max(1, min(top, 100)))
    try:
        rows = ctx.conn.execute(sql, params).fetchall()
    except Exception:
        return []
    return [_message_dict(r) for r in rows]


def get_stats(
    ctx: ToolContext,
    metric: str,
    chat_id: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    bucket: str = "month",
    top: int = 25,
    session_gap: int = 3600,
) -> dict:
    conn = ctx.conn
    if metric == "per_sender":
        return {"metric": metric, "rows": [asdict(s) for s in stats_mod.per_sender(conn, chat_id, date_from, date_to)]}
    if metric == "response_time":
        return {
            "metric": metric,
            "rows": [asdict(r) for r in stats_mod.response_times(conn, chat_id, date_from, date_to, session_gap_seconds=session_gap)],
            "note": "any=every consecutive pair; reply=sender change; reply-session=sender change within session gap",
        }
    if metric == "words":
        return {"metric": metric, "rows": [{"word": w, "count": n} for w, n in stats_mod.word_frequency(conn, chat_id, date_from, date_to, top=top)]}
    if metric == "emojis":
        return {"metric": metric, "rows": [{"emoji": e, "count": n} for e, n in stats_mod.emoji_frequency(conn, chat_id, date_from, date_to, top=top)]}
    if metric == "volume":
        return {"metric": metric, "bucket": bucket, "rows": [{"period": p, "count": n} for p, n in stats_mod.volume(conn, chat_id, date_from, date_to, bucket)]}
    if metric == "time_of_day":
        data = stats_mod.time_of_day(conn, chat_id, date_from, date_to)
        return {"metric": metric, "hours": [{"hour": h, "count": n} for h, n in data["hours"]], "weekdays": [{"weekday": d, "count": n} for d, n in data["weekdays"]]}
    if metric == "sessions":
        sessions = stats_mod.sessions(conn, chat_id, date_from, date_to, gap_seconds=session_gap)
        starters: dict[str, int] = {}
        for s in sessions:
            starters[s.starter] = starters.get(s.starter, 0) + 1
        return {
            "metric": metric,
            "session_count": len(sessions),
            "starters": sorted(({"name": k, "sessions": v} for k, v in starters.items()), key=lambda x: -x["sessions"]),
            "longest": [
                {"start": s.start_local, "end": s.end_local, "messages": s.messages, "starter": s.starter}
                for s in sorted(sessions, key=lambda s: s.messages, reverse=True)[:5]
            ],
        }
    return {"error": f"unknown metric {metric!r}", "valid": ["per_sender", "response_time", "words", "emojis", "volume", "time_of_day", "sessions"]}


def get_context(ctx: ToolContext, message_id: str, before: int = 5, after: int = 5) -> dict:
    anchor = ctx.conn.execute(
        "SELECT chat_id, ts FROM messages WHERE id = ?", (message_id,)
    ).fetchone()
    if anchor is None:
        return {"error": f"message {message_id!r} not found", "messages": []}
    before = max(0, min(before, 25))
    after = max(0, min(after, 25))
    prev = ctx.conn.execute(
        "SELECT id, local_date, local_time, sender_id, text, msg_type FROM messages "
        "WHERE chat_id = ? AND ts <= ? ORDER BY ts DESC, rowid DESC LIMIT ?",
        (anchor["chat_id"], anchor["ts"], before + 1),
    ).fetchall()
    nxt = ctx.conn.execute(
        "SELECT id, local_date, local_time, sender_id, text, msg_type FROM messages "
        "WHERE chat_id = ? AND ts > ? ORDER BY ts ASC, rowid ASC LIMIT ?",
        (anchor["chat_id"], anchor["ts"], after),
    ).fetchall()
    ordered = list(reversed(prev)) + list(nxt)
    return {"messages": [_message_dict(r) for r in ordered], "anchor": message_id}


def topic_clusters(
    ctx: ToolContext,
    chat_id: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    min_cluster_size: int = 5,
    top_terms: int = 8,
    top_examples: int = 3,
    evolution_bucket: str | None = None,
) -> dict:
    result, evolution = clustering_mod.analyze_topics(
        ctx.collection,
        chat_id=chat_id,
        date_from=date_from,
        date_to=date_to,
        min_cluster_size=max(2, min(min_cluster_size, 50)),
        top_terms=top_terms,
        top_examples=top_examples,
        evolution_bucket=evolution_bucket,
    )
    return {
        "total_windows": result.total,
        "noise": result.noise,
        "noise_ratio": round(result.noise_ratio, 3),
        "clusters": [
            {
                "cluster_id": c.cluster_id,
                "size": c.size,
                "label": c.label,
                "terms": c.terms,
                "start_date": c.start_date,
                "end_date": c.end_date,
                "examples": _cluster_examples(ctx, c.examples),
            }
            for c in result.clusters
        ],
        "evolution": evolution,
    }


def _cluster_examples(ctx: ToolContext, examples, limit: int = 10) -> list[dict]:
    """Turn window-level examples into real, citable messages."""
    out: list[dict] = []
    for e in examples:
        messages = _messages_by_ids(ctx.conn, e.message_ids)[:limit]
        if messages:
            first = messages[0]
            out.append(
                {
                    "id": first["id"],
                    "window_id": e.window_id or e.id,
                    "date": first["date"],
                    "sender": first["sender"],
                    "text": first["text"],
                    "messages": messages,
                }
            )
        else:
            out.append(
                {
                    "id": e.id,
                    "window_id": e.window_id or e.id,
                    "date": e.date,
                    "sender": e.sender,
                    "text": _clip(e.text, 300),
                    "messages": [],
                }
            )
    return out


def inside_joke_candidates(
    ctx: ToolContext,
    chat_id: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    min_count: int = 5,
    top: int = 30,
) -> dict:
    candidates = inside_jokes.candidate_ngrams(
        ctx.conn, chat_id=chat_id, date_from=date_from, date_to=date_to,
        min_count=max(2, min_count), top=min(top, 100),
    )
    return {
        "candidates": [
            {
                "phrase": c.phrase,
                "count": c.count,
                "first": c.first,
                "last": c.last,
                "span_days": c.span_days,
                "senders": [{"name": n, "count": k} for n, k in c.senders],
            }
            for c in candidates
        ]
    }


def phrase_timeline(
    ctx: ToolContext,
    phrase: str,
    chat_id: str | None = None,
    bucket: str = "month",
    top: int = 15,
) -> dict:
    timeline = inside_jokes.phrase_timeline(ctx.conn, phrase, chat_id=chat_id, bucket=bucket)
    evidence = inside_jokes.phrase_evidence(ctx.conn, phrase, top=top, chat_id=chat_id)
    return {
        "phrase": phrase,
        "total": sum(n for _, n in timeline),
        "timeline": [{"period": p, "count": n} for p, n in timeline],
        "evidence": evidence,
    }


def list_chats(ctx: ToolContext) -> list[dict]:
    rows = ctx.conn.execute(
        """
        SELECT c.chat_id, c.name, c.is_group, c.first_ts, c.last_ts,
               (SELECT COUNT(*) FROM messages m WHERE m.chat_id = c.chat_id) AS n
        FROM chats c ORDER BY n DESC
        """
    ).fetchall()
    return [
        {
            "chat_id": r["chat_id"],
            "name": r["name"],
            "is_group": bool(r["is_group"]),
            "messages": r["n"],
            "first": _date(r["first_ts"]),
            "last": _date(r["last_ts"]),
        }
        for r in rows
    ]


def _date(ts: int | None) -> str:
    if not ts:
        return ""
    from datetime import datetime

    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")


# --- schema + dispatch -----------------------------------------------------

_FILTERS = {
    "chat_id": {"type": "string", "description": "Restrict to one chat_id (see list_chats)"},
    "date_from": {"type": "string", "description": "Start date YYYY-MM-DD (inclusive)"},
    "date_to": {"type": "string", "description": "End date YYYY-MM-DD (inclusive)"},
}

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "list_chats",
            "description": "List available chats with their id, message count and date range.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "semantic_search",
            "description": "Find messages by meaning/topic. Best for open questions, themes, inside jokes, how something was discussed. Returns messages with id/date/sender/text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top": {"type": "integer", "description": "How many results (default 8)"},
                    "sender": {"type": "string", "description": "Filter by sender name"},
                    "context": {"type": "integer", "description": "Also include this many real messages before/after each hit (default 0)"},
                    **_FILTERS,
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "hybrid_search",
            "description": "Preferred default for factual lookups: merges semantic meaning and exact keyword search (RRF), best recall.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top": {"type": "integer", "description": "How many fused results (default 10)"},
                    "sender": {"type": "string", "description": "Filter by sender name"},
                    "context": {"type": "integer", "description": "Also include this many real messages before/after each hit (default 0)"},
                    **_FILTERS,
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "smart_search",
            "description": "Highest-quality search: the model expands the query into paraphrases (and optionally drafts a likely answer) and fuses semantic + keyword results. Best for broad, vague or hard questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top": {"type": "integer", "description": "How many results (default 10)"},
                    "context": {"type": "integer", "description": "Real messages before/after each hit (default 2)"},
                    "expand": {"type": "integer", "description": "How many paraphrases to generate (default 3)"},
                    "use_hyde": {"type": "boolean", "description": "Also embed a hypothetical answer"},
                    "sender": {"type": "string", "description": "Filter by sender name"},
                    **_FILTERS,
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "window_search",
            "description": "Search whole conversation windows (blocks of consecutive messages) and get the real messages inside each hit. Best for richer context, how a discussion developed, or vague topic queries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top": {"type": "integer", "description": "How many windows (default 5)"},
                    **_FILTERS,
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "keyword_search",
            "description": "Exact word/phrase search (FTS). Best for specific names, nicknames or phrases the model knows literally.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top": {"type": "integer"},
                    **_FILTERS,
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stats",
            "description": "Computed statistics. Use for counts, averages, trends over time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {
                        "type": "string",
                        "enum": ["per_sender", "response_time", "words", "emojis", "volume", "time_of_day", "sessions"],
                    },
                    "bucket": {"type": "string", "enum": ["day", "week", "month", "year"]},
                    "top": {"type": "integer"},
                    **_FILTERS,
                },
                "required": ["metric"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_context",
            "description": "Show the messages surrounding a given message id (to see how a conversation unfolded).",
            "parameters": {
                "type": "object",
                "properties": {
                    "message_id": {"type": "string"},
                    "before": {"type": "integer"},
                    "after": {"type": "integer"},
                },
                "required": ["message_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "topic_clusters",
            "description": "Discover recurring topics/themes by clustering conversation windows. Use for 'what do we talk about most', topic changes over time. Returns clusters with distinctive terms and example message ids.",
            "parameters": {
                "type": "object",
                "properties": {
                    "min_cluster_size": {"type": "integer", "description": "Minimum messages per topic (default 5)"},
                    "evolution_bucket": {"type": "string", "enum": ["month", "year"], "description": "Also return how topics evolve over time"},
                    **_FILTERS,
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inside_joke_candidates",
            "description": "Find frequently repeated phrases (2-4 words), i.e. candidate inside jokes. Returns phrase, frequency, first/last date and main users.",
            "parameters": {
                "type": "object",
                "properties": {
                    "min_count": {"type": "integer", "description": "Minimum occurrences (default 5)"},
                    "top": {"type": "integer"},
                    **_FILTERS,
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "phrase_timeline",
            "description": "How often an exact phrase/nickname recurred over time, with example messages and ids. Use to see if an inside joke faded or changed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "phrase": {"type": "string"},
                    "bucket": {"type": "string", "enum": ["month", "year"]},
                    "top": {"type": "integer"},
                    **_FILTERS,
                },
                "required": ["phrase"],
            },
        },
    },
]

Dispatch = Callable[[ToolContext, dict], Any]


def _call_semantic(ctx: ToolContext, args: dict) -> Any:
    args = {k: v for k, v in args.items() if v is not None}
    return semantic_search(ctx, **args)


def _call_hybrid(ctx: ToolContext, args: dict) -> Any:
    args = {k: v for k, v in args.items() if v is not None}
    return hybrid_search(ctx, **args)


def _call_smart(ctx: ToolContext, args: dict) -> Any:
    args = {k: v for k, v in args.items() if v is not None}
    return smart_search(ctx, **args)


def _call_window(ctx: ToolContext, args: dict) -> Any:
    args = {k: v for k, v in args.items() if v is not None}
    return window_search(ctx, **args)


def _call_keyword(ctx: ToolContext, args: dict) -> Any:
    args = {k: v for k, v in args.items() if v is not None}
    return keyword_search(ctx, **args)


def _call_stats(ctx: ToolContext, args: dict) -> Any:
    args = {k: v for k, v in args.items() if v is not None}
    return get_stats(ctx, **args)


def _call_context(ctx: ToolContext, args: dict) -> Any:
    args = {k: v for k, v in args.items() if v is not None}
    return get_context(ctx, **args)


def _call_list(ctx: ToolContext, args: dict) -> Any:
    return list_chats(ctx)


def _call_topics(ctx: ToolContext, args: dict) -> Any:
    args = {k: v for k, v in args.items() if v is not None}
    return topic_clusters(ctx, **args)


def _call_jokes(ctx: ToolContext, args: dict) -> Any:
    args = {k: v for k, v in args.items() if v is not None}
    return inside_joke_candidates(ctx, **args)


def _call_phrase(ctx: ToolContext, args: dict) -> Any:
    args = {k: v for k, v in args.items() if v is not None}
    return phrase_timeline(ctx, **args)


_DISPATCH: dict[str, Dispatch] = {
    "list_chats": _call_list,
    "semantic_search": _call_semantic,
    "hybrid_search": _call_hybrid,
    "smart_search": _call_smart,
    "window_search": _call_window,
    "keyword_search": _call_keyword,
    "get_stats": _call_stats,
    "get_context": _call_context,
    "topic_clusters": _call_topics,
    "inside_joke_candidates": _call_jokes,
    "phrase_timeline": _call_phrase,
}


def dispatch(ctx: ToolContext, name: str, arguments: dict) -> Any:
    fn = _DISPATCH.get(name)
    if fn is None:
        return {"error": f"unknown tool {name!r}"}
    try:
        return fn(ctx, arguments or {})
    except TypeError as exc:
        return {"error": f"bad arguments for {name}: {exc}"}
    except Exception as exc:  # noqa: BLE001 - surface tool errors to the model
        return {"error": f"{type(exc).__name__}: {exc}"}


def collect_message_ids(result: Any, into: dict[str, dict]) -> None:
    """Recursively collect every message record (dict with an 'id') from a tool result."""
    if isinstance(result, dict):
        if "id" in result and isinstance(result["id"], str) and "text" in result:
            into.setdefault(result["id"], result)
        for value in result.values():
            collect_message_ids(value, into)
    elif isinstance(result, list):
        for item in result:
            collect_message_ids(item, into)


def dumps(result: Any) -> str:
    return json.dumps(result, ensure_ascii=False, default=_records_to_dump)
