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

from ..db import stats as stats_mod

MAX_TEXT = 400

# --- context ---------------------------------------------------------------


@dataclass
class ToolContext:
    conn: Any
    collection: Any
    embedder: Any


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


# --- tool implementations --------------------------------------------------


def semantic_search(
    ctx: ToolContext,
    query: str,
    top: int = 8,
    chat_id: str | None = None,
    sender: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
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
]

Dispatch = Callable[[ToolContext, dict], Any]


def _call_semantic(ctx: ToolContext, args: dict) -> Any:
    args = {k: v for k, v in args.items() if v is not None}
    return semantic_search(ctx, **args)


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


_DISPATCH: dict[str, Dispatch] = {
    "list_chats": _call_list,
    "semantic_search": _call_semantic,
    "keyword_search": _call_keyword,
    "get_stats": _call_stats,
    "get_context": _call_context,
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
