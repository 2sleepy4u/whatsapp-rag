"""Statistics engine over the messages table.

All functions take an open sqlite connection and plain keyword filters, and
return lightweight dataclasses / dicts so they can be reused by the CLI today
and by the RAG tools in Phase 4.

Filtering is done on ``local_date`` (``YYYY-MM-DD``) so date ranges are
timezone-correct without extra arithmetic. Service traffic (system/call) is
excluded by default; deleted and placeholder rows are excluded from text-based
metrics automatically.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from dataclasses import dataclass, field

import emoji as emoji_lib

# Message types that never represent a real human utterance.
DEFAULT_EXCLUDE = ("system", "call", "deleted")
# Types that carry no analysable text for word/emoji metrics.
TEXT_TYPES = ("text", "edited")

_WORD_RE = re.compile(r"[0-9a-zàáèéìíòóùúâêîôûäëïöüç']+", re.IGNORECASE)
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

ITALIAN_STOPWORDS = {
    "a", "ad", "al", "alla", "allo", "ai", "agli", "alle", "anche", "ancora",
    "avete", "aveva", "avevo", "bene", "che", "chi", "ci", "cio", "cioe",
    "come", "con", "cosa", "cosi", "da", "dal", "dalla", "dei", "del", "della",
    "delle", "di", "dove", "e", "ed", "era", "erano", "essere", "fa", "fai",
    "fare", "gli", "ha", "hai", "hanno", "ho", "i", "il", "in", "io", "la",
    "le", "lei", "lo", "loro", "lui", "ma", "me", "mi", "mia", "mie", "miei",
    "mio", "ne", "negli", "nel", "nella", "no", "noi", "non", "o", "per",
    "perche", "piu", "più", "puo", "puoi", "qua", "quando", "quello", "questa",
    "questo", "se", "sei", "senza", "si", "siamo", "sono", "sta", "stai",
    "su", "sua", "sue", "sul", "sulla", "suo", "suoi", "te", "ti", "tra",
    "tu", "tua", "tue", "tuo", "tuoi", "un", "una", "uno", "va", "vi", "voi",
    "è", "e'", "ok", "okay", "the", "and", "you", "lol", "ahah", "ahaha",
    "haha", "hahaha", "ahahah", "ahahaha", "ahahahah", "ahahahaha", "cmq",
    "cmq", "nn", "xk", "xkè", "xké", "poi", "già", "gia", "giusto", "tipo",
}


def _filters(
    chat_id: str | None,
    date_from: str | None,
    date_to: str | None,
    sender: str | None = None,
    exclude: tuple[str, ...] = DEFAULT_EXCLUDE,
    types: tuple[str, ...] | None = None,
) -> tuple[str, list]:
    clauses: list[str] = []
    params: list = []
    if chat_id:
        clauses.append("chat_id = ?")
        params.append(chat_id)
    if sender:
        clauses.append("sender_id LIKE ?")
        params.append(f"%::{sender}")
    if date_from:
        clauses.append("local_date >= ?")
        params.append(date_from)
    if date_to:
        clauses.append("local_date <= ?")
        params.append(date_to)
    if types is not None:
        clauses.append("msg_type IN (%s)" % ",".join("?" * len(types)))
        params.extend(types)
    elif exclude:
        clauses.append("msg_type NOT IN (%s)" % ",".join("?" * len(exclude)))
        params.extend(exclude)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    return where, params


@dataclass
class SenderStat:
    name: str
    messages: int
    share: float
    chars: int
    words: int
    first_local: str
    last_local: str

    @property
    def avg_chars(self) -> float:
        return self.chars / self.messages if self.messages else 0.0


def per_sender(
    conn,
    chat_id: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[SenderStat]:
    where, params = _filters(chat_id, date_from, date_to, types=TEXT_TYPES)
    rows = conn.execute(
        f"""
        SELECT sender_id, text, local_date FROM messages
        {where} AND sender_id IS NOT NULL
        """,
        params,
    ).fetchall()
    total = len(rows)
    agg: dict[str, dict] = {}
    for r in rows:
        sid = r["sender_id"]
        name = sid.split("::", 1)[1] if "::" in sid else sid
        a = agg.setdefault(
            name, {"messages": 0, "chars": 0, "words": 0, "first": r["local_date"], "last": r["local_date"]}
        )
        a["messages"] += 1
        a["chars"] += len(r["text"])
        a["words"] += len(_WORD_RE.findall(r["text"]))
        a["first"] = min(a["first"], r["local_date"])
        a["last"] = max(a["last"], r["local_date"])
    out = [
        SenderStat(
            name=name,
            messages=a["messages"],
            share=a["messages"] / total if total else 0.0,
            chars=a["chars"],
            words=a["words"],
            first_local=a["first"],
            last_local=a["last"],
        )
        for name, a in agg.items()
    ]
    out.sort(key=lambda s: s.messages, reverse=True)
    return out


@dataclass
class ResponseTimeStat:
    variant: str
    count: int
    mean_sec: float
    median_sec: float
    p90_sec: float
    min_sec: float
    max_sec: float

    def human(self, value: float | None = None) -> str:
        return _human_seconds(self.median_sec if value is None else value)


def _human_seconds(sec: float) -> str:
    if sec < 60:
        return f"{sec:.0f}s"
    if sec < 3600:
        return f"{sec / 60:.1f}m"
    if sec < 86400:
        return f"{sec / 3600:.1f}h"
    return f"{sec / 86400:.1f}d"


def response_times(
    conn,
    chat_id: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    session_gap_seconds: int = 3600,
) -> list[ResponseTimeStat]:
    where, params = _filters(chat_id, date_from, date_to, types=None, exclude=("system", "call"))
    rows = conn.execute(
        f"""
        WITH ordered AS (
            SELECT chat_id, ts, sender_id,
                   LAG(ts) OVER w AS prev_ts,
                   LAG(sender_id) OVER w AS prev_sender
            FROM messages
            {where}
            WINDOW w AS (PARTITION BY chat_id ORDER BY ts, rowid)
        )
        SELECT ts, sender_id, prev_ts, prev_sender
        FROM ordered
        WHERE prev_ts IS NOT NULL
        """,
        params,
    ).fetchall()

    any_gaps: list[float] = []
    reply_gaps: list[float] = []
    reply_session: list[float] = []
    for r in rows:
        gap = r["ts"] - r["prev_ts"]
        if gap < 0:
            continue
        any_gaps.append(gap)
        if r["sender_id"] != r["prev_sender"]:
            reply_gaps.append(gap)
            if gap <= session_gap_seconds:
                reply_session.append(gap)

    return [
        _summarize("any", any_gaps),
        _summarize("reply", reply_gaps),
        _summarize("reply-session", reply_session),
    ]


def _summarize(variant: str, gaps: list[float]) -> ResponseTimeStat:
    if not gaps:
        return ResponseTimeStat(variant, 0, 0, 0, 0, 0, 0)
    ordered = sorted(gaps)
    p90 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))]
    return ResponseTimeStat(
        variant=variant,
        count=len(gaps),
        mean_sec=statistics.fmean(gaps),
        median_sec=statistics.median(gaps),
        p90_sec=float(p90),
        min_sec=float(min(gaps)),
        max_sec=float(max(gaps)),
    )


def word_frequency(
    conn,
    chat_id: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    top: int = 40,
    min_len: int = 3,
    stopwords: bool = True,
) -> list[tuple[str, int]]:
    where, params = _filters(chat_id, date_from, date_to, types=TEXT_TYPES)
    counter: Counter[str] = Counter()
    for (text,) in conn.execute(f"SELECT text FROM messages {where}", params):
        text = _URL_RE.sub(" ", text)
        for word in _WORD_RE.findall(text.lower()):
            if len(word) < min_len:
                continue
            if stopwords and word in ITALIAN_STOPWORDS:
                continue
            counter[word] += 1
    return counter.most_common(top)


def emoji_frequency(
    conn,
    chat_id: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    top: int = 30,
) -> list[tuple[str, int]]:
    where, params = _filters(chat_id, date_from, date_to, types=None, exclude=("system", "call"))
    counter: Counter[str] = Counter()
    for (text,) in conn.execute(f"SELECT text FROM messages {where}", params):
        for item in emoji_lib.emoji_list(text):
            counter[item["emoji"]] += 1
    return counter.most_common(top)


@dataclass
class Session:
    chat_id: str
    start_ts: int
    end_ts: int
    start_local: str
    end_local: str
    messages: int
    starter: str
    gap_before_sec: int | None = None

    @property
    def duration_sec(self) -> int:
        return self.end_ts - self.start_ts


def sessions(
    conn,
    chat_id: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    gap_seconds: int = 3600,
) -> list[Session]:
    where, params = _filters(chat_id, date_from, date_to, types=None, exclude=("system", "call"))
    rows = conn.execute(
        f"SELECT chat_id, ts, ts_local, sender_id FROM messages {where} ORDER BY chat_id, ts, rowid",
        params,
    ).fetchall()

    result: list[Session] = []
    cur: Session | None = None
    for r in rows:
        sid = r["sender_id"] or ""
        name = sid.split("::", 1)[1] if "::" in sid else sid
        if cur is None or r["chat_id"] != cur.chat_id or r["ts"] - cur.end_ts > gap_seconds:
            if cur is not None:
                result.append(cur)
            cur = Session(
                chat_id=r["chat_id"],
                start_ts=r["ts"],
                end_ts=r["ts"],
                start_local=r["ts_local"],
                end_local=r["ts_local"],
                messages=1,
                starter=name,
                gap_before_sec=None if not result else r["ts"] - result[-1].end_ts,
            )
        else:
            cur.end_ts = r["ts"]
            cur.end_local = r["ts_local"]
            cur.messages += 1
    if cur is not None:
        result.append(cur)
    return result


def volume(
    conn,
    chat_id: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    bucket: str = "month",
) -> list[tuple[str, int]]:
    fmt = {"day": "%Y-%m-%d", "week": "%Y-W%W", "month": "%Y-%m", "year": "%Y"}[bucket]
    where, params = _filters(chat_id, date_from, date_to, exclude=("system",))
    rows = conn.execute(
        f"""
        SELECT strftime('{fmt}', local_date) AS bucket, COUNT(*) n
        FROM messages {where}
        GROUP BY bucket ORDER BY bucket
        """,
        params,
    ).fetchall()
    return [(r["bucket"], r["n"]) for r in rows]


def time_of_day(
    conn,
    chat_id: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> dict[str, list[tuple[int, int]]]:
    where, params = _filters(chat_id, date_from, date_to, exclude=("system",))
    hours = conn.execute(
        f"""
        SELECT CAST(substr(local_time, 1, 2) AS INTEGER) AS h, COUNT(*) n
        FROM messages {where}
        GROUP BY h ORDER BY h
        """,
        params,
    ).fetchall()
    weekdays = conn.execute(
        f"""
        SELECT CAST(strftime('%w', local_date) AS INTEGER) AS d, COUNT(*) n
        FROM messages {where}
        GROUP BY d ORDER BY d
        """,
        params,
    ).fetchall()
    return {
        "hours": [(r["h"], r["n"]) for r in hours],
        "weekdays": [(r["d"], r["n"]) for r in weekdays],
    }
