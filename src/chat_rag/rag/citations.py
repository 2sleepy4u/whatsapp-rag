from __future__ import annotations

import re
from dataclasses import dataclass

# Message ids are a 16-char sha1 prefix, optionally suffixed with -N for
# duplicate identical messages.
ID_RE = re.compile(r"\b([0-9a-f]{16}(?:-\d+)?)\b")


@dataclass
class Citation:
    id: str
    date: str
    time: str
    sender: str
    text: str
    found: bool = True


def extract_ids(text: str) -> list[str]:
    seen: dict[str, None] = {}
    for match in ID_RE.findall(text or ""):
        seen.setdefault(match, None)
    return list(seen)


def get_message(conn, message_id: str) -> Citation | None:
    row = conn.execute(
        "SELECT id, local_date, local_time, sender_id, text FROM messages WHERE id = ?",
        (message_id,),
    ).fetchone()
    if row is None:
        return None
    sender = row["sender_id"].split("::", 1)[1] if row["sender_id"] and "::" in row["sender_id"] else (row["sender_id"] or "")
    return Citation(
        id=row["id"],
        date=row["local_date"],
        time=row["local_time"],
        sender=sender,
        text=row["text"],
    )


def resolve(conn, ids: list[str]) -> list[Citation]:
    out: list[Citation] = []
    for mid in ids:
        cit = get_message(conn, mid)
        if cit is None:
            out.append(Citation(id=mid, date="", time="", sender="", text="", found=False))
        else:
            out.append(cit)
    return out


def resolve_answer(conn, answer: str, fallback_ids: list[str]) -> list[Citation]:
    """Prefer ids cited in the answer; fall back to ids surfaced by tools."""
    ids = extract_ids(answer)
    if not ids:
        ids = fallback_ids
    return resolve(conn, ids)
