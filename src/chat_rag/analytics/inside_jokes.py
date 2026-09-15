"""Inside-joke discovery.

Approach: count repeated n-grams (2–4 words) across the chat, rank by frequency,
and report each candidate's first/last occurrence and who used it most. A
separate ``phrase_timeline`` shows how often a given phrase recurred over time,
and ``phrase_evidence`` returns the matching messages (with ids) for citations.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime

from ..db.stats import ITALIAN_STOPWORDS

_WORD_RE = re.compile(r"[0-9a-zàáèéìíòóùúâêîôûäëïöüç']+", re.IGNORECASE)


@dataclass
class JokeCandidate:
    phrase: str
    count: int
    first: str
    last: str
    senders: list[tuple[str, int]]

    @property
    def span_days(self) -> int:
        try:
            return (datetime.strptime(self.last, "%Y-%m-%d") - datetime.strptime(self.first, "%Y-%m-%d")).days
        except ValueError:
            return 0


def _tokens(text: str) -> list[str]:
    return [w for w in _WORD_RE.findall(text.lower()) if len(w) >= 2]


def _meaningful(tokens: tuple[str, ...]) -> bool:
    content = [t for t in tokens if t not in ITALIAN_STOPWORDS and len(t) >= 3]
    return len(content) >= 1


def _sender_name(sender_id: str | None) -> str:
    if not sender_id:
        return ""
    return sender_id.split("::", 1)[1] if "::" in sender_id else sender_id


def candidate_ngrams(
    conn,
    chat_id: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    min_count: int = 5,
    max_n: int = 4,
    top: int = 50,
) -> list[JokeCandidate]:
    clauses = ["msg_type IN ('text','edited')"]
    params: list = []
    if chat_id:
        clauses.append("chat_id = ?")
        params.append(chat_id)
    if date_from:
        clauses.append("local_date >= ?")
        params.append(date_from)
    if date_to:
        clauses.append("local_date <= ?")
        params.append(date_to)
    sql = (
        "SELECT text, local_date, sender_id FROM messages "
        f"WHERE {' AND '.join(clauses)}"
    )

    stats: dict[tuple[str, ...], dict] = {}
    for text, day, sender_id in conn.execute(sql, params):
        tokens = _tokens(text)
        if len(tokens) < 2:
            continue
        if len(tokens) > 40:
            tokens = tokens[:40]
        name = _sender_name(sender_id)
        for n in range(2, max_n + 1):
            if len(tokens) < n:
                break
            for i in range(len(tokens) - n + 1):
                gram = tuple(tokens[i : i + n])
                if not _meaningful(gram):
                    continue
                entry = stats.get(gram)
                if entry is None:
                    entry = {"count": 0, "first": day, "last": day, "senders": Counter()}
                    stats[gram] = entry
                entry["count"] += 1
                entry["first"] = min(entry["first"], day)
                entry["last"] = max(entry["last"], day)
                if name:
                    entry["senders"][name] += 1

    candidates = [
        JokeCandidate(
            phrase=" ".join(gram),
            count=e["count"],
            first=e["first"],
            last=e["last"],
            senders=e["senders"].most_common(5),
        )
        for gram, e in stats.items()
        if e["count"] >= min_count
    ]
    candidates.sort(key=lambda c: (c.count, len(c.phrase.split())), reverse=True)
    return _suppress_overlaps(candidates)[:top]


def _contains(long_tokens: tuple[str, ...], short_tokens: tuple[str, ...]) -> bool:
    n, m = len(long_tokens), len(short_tokens)
    if m > n:
        return False
    return any(long_tokens[i : i + m] == short_tokens for i in range(n - m + 1))


def _suppress_overlaps(candidates: list[JokeCandidate]) -> list[JokeCandidate]:
    """Drop a phrase when a longer, at least as frequent phrase contains it."""
    accepted: list[JokeCandidate] = []
    accepted_tokens: list[tuple[tuple[str, ...], int]] = []
    for cand in candidates:
        tokens = tuple(cand.phrase.split())
        if any(count >= cand.count and _contains(other, tokens) for other, count in accepted_tokens):
            continue
        accepted.append(cand)
        accepted_tokens.append((tokens, cand.count))
    return accepted


def _fts_phrase(phrase: str) -> str:
    tokens = [t for t in _WORD_RE.findall(phrase) if t]
    if not tokens:
        return '""'
    return '"' + " ".join(tokens) + '"'


def phrase_evidence(
    conn,
    phrase: str,
    top: int = 15,
    chat_id: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[dict]:
    clauses: list[str] = []
    params: list = [_fts_phrase(phrase)]
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
        FROM messages_fts f JOIN messages m ON m.rowid = f.rowid
        WHERE messages_fts MATCH ?{extra}
        ORDER BY m.ts
        LIMIT ?
    """
    params.append(max(1, min(top, 100)))
    rows = conn.execute(sql, params).fetchall()
    return [
        {
            "id": r["id"],
            "date": r["local_date"],
            "time": r["local_time"],
            "sender": _sender_name(r["sender_id"]),
            "text": r["text"][:300],
        }
        for r in rows
    ]


def phrase_timeline(
    conn,
    phrase: str,
    chat_id: str | None = None,
    bucket: str = "month",
) -> list[tuple[str, int]]:
    fmt = {"day": "%Y-%m-%d", "week": "%Y-%W", "month": "%Y-%m", "year": "%Y"}[bucket]
    clauses: list[str] = []
    params: list = [_fts_phrase(phrase)]
    if chat_id:
        clauses.append("m.chat_id = ?")
        params.append(chat_id)
    extra = (" AND " + " AND ".join(clauses)) if clauses else ""
    sql = f"""
        SELECT strftime('{fmt}', m.local_date) AS period, COUNT(*) n
        FROM messages_fts f JOIN messages m ON m.rowid = f.rowid
        WHERE messages_fts MATCH ?{extra}
        GROUP BY period ORDER BY period
    """
    rows = conn.execute(sql, params).fetchall()
    return [(r["period"], r["n"]) for r in rows]
