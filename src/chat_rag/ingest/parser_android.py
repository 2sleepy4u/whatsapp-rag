"""Parser for Android WhatsApp chat exports.

Android exports look like::

    17/04/2023, 14:03 - Marco: Ciao
    17/04/2023, 14:05 - Me: Ehi
    continua qui
    17/04/23, 14:07 - Messaggi e chiamate sono protetti con la crittografia end-to-end.

Key challenges handled here:

* two line formats (``DD/MM/YY`` and ``DD/MM/YYYY``), 24h and 12h clocks;
* multi-line messages (continuation lines have no timestamp prefix);
* sender names that may contain ``:`` — we split on the first ``: `` only;
* system / call / encryption notices and media placeholders;
* edited and deleted markers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator, Literal

from zoneinfo import ZoneInfo

MsgType = Literal[
    "text",
    "system",
    "call",
    "voice",
    "image",
    "video",
    "document",
    "sticker",
    "gif",
    "contact",
    "location",
    "deleted",
    "edited",
    "media",
]

# 17/04/2023, 14:03 - rest
# 17/04/23, 2:03 pm - rest
# 17/04/2023, 14:03:12 - rest
_LINE_RE = re.compile(
    r"^(?P<date>\d{1,2}/\d{1,2}/\d{2,4}),?\s+"
    r"(?P<time>\d{1,2}:\d{2}(?::\d{2})?)\s*"
    r"(?P<ampm>[AaPp]\.?\s?[Mm]\.?)?\s*"
    r"[-–]\s*"
    r"(?P<rest>.*)$"
)

_DATE_FORMATS = ("%d/%m/%Y", "%d/%m/%y")
_TIME_FORMATS = ("%H:%M:%S", "%H:%M", "%I:%M:%S %p", "%I:%M %p")

# System / service notices (Italian Android exports, plus common English).
_SYSTEM_PATTERNS = [
    r"crittografia end-to-end",
    r"protetti con la crittografia",
    r"end-to-end encrypted",
    r"ha cambiato l'oggetto",
    r"cambiato l'oggetto",
    r"changed the subject",
    r"ha cambiato l'icona",
    r"changed this group's icon",
    r"ha cambiato il numero",
    r"changed their phone number",
    r"ha cambiato il suo numero",
    r"ha aggiunto",
    r"hanno aggiunto",
    r"added you",
    r"ti ha aggiunto",
    r"added ",
    r"è entrato",
    r"e' entrato",
    r"si è unito",
    r"joined using",
    r"è uscito",
    r"e' uscito",
    r"left$",
    r"ha rimosso",
    r"removed ",
    r"ha creato (il gruppo|questo gruppo)",
    r"created (group|this group)",
    r"hai creato",
    r"ha inserito",
    r"ha modificato la descrizione",
    r"changed the group description",
    r"messaggi temporanei",
    r"messages are set to disappear",
    r"disappearing messages",
    r"codice di sicurezza",
    r"security code (changed|with)",
    r"la tua richiesta di sicurezza",
    r"ha bloccato",
    r"blocked ",
    r"non è più un membro",
    r"non e' piu' un membro",
    r"waiting for this message",
    r"in attesa di questo messaggio",
    r"non hai ricevuto questo messaggio",
    r"you received this message",
    r"ha avviato una",
    r"crittografia dei messaggi",
    r"changed their phone",
    r"cambiato il numero di telefono",
]
_SYSTEM_RE = re.compile("|".join(_SYSTEM_PATTERNS), re.IGNORECASE)

_CALL_RE = re.compile(
    r"^(?:chiamata|videochiamata|voice call|video call|missed voice call|"
    r"missed video call|hai perso una|hai effettuato una|ha effettuato una|"
    r"ha avviato una|call ended|chiamata senza risposta)",
    re.IGNORECASE,
)

_DELETED_RE = re.compile(
    r"(questo messaggio è stato eliminato|questo messaggio e stato eliminato|"
    r"this message was deleted|messaggio eliminato)",
    re.IGNORECASE,
)
_EDITED_RE = re.compile(
    r"\s*[<\(]?(questo messaggio è stato modificato|this message was edited)"
    r"[>\)]?\s*$",
    re.IGNORECASE,
)

_MEDIA_OMITTED_RE = re.compile(
    r"^<(?P<kind>media|allegato|image|video|audio|document|gif|sticker|contact|"
    r"posizione|media omesso|media omessi|allegato omesso|allegati omessi)"
    r"\s*(?:omess[oi])?>$",
    re.IGNORECASE,
)
_FILE_ATTACHED_RE = re.compile(
    r"(?P<name>(?:IMG|VID|AUD|PTT|STK|DOC|GIF|PHOTO|VIDEO)-\d{8}-WA\d+\.[A-Za-z0-9]+)"
    r"\s*\((?:file allegato|file attached|allegato)\)",
    re.IGNORECASE,
)

_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".gif"}
_VIDEO_EXT = {".mp4", ".mov", ".3gp", ".mkv", ".webm"}
_VOICE_EXT = {".opus", ".ogg", ".m4a", ".aac", ".amr"}
_DOC_EXT = {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".txt", ".zip"}


@dataclass
class ParsedMessage:
    ts: int
    ts_local: str
    local_date: str
    local_time: str
    sender: str | None
    text: str
    raw_line: str
    msg_type: MsgType
    media_ref: str | None = None
    line_no: int = 0


@dataclass
class ParseStats:
    lines: int = 0
    messages: int = 0
    continuations: int = 0
    by_type: dict[str, int] = field(default_factory=dict)

    def bump(self, msg_type: str) -> None:
        self.by_type[msg_type] = self.by_type.get(msg_type, 0) + 1


def _parse_datetime(date_s: str, time_s: str, ampm: str | None) -> datetime:
    for fmt in _DATE_FORMATS:
        try:
            date_part = datetime.strptime(date_s, fmt)
            break
        except ValueError:
            continue
    else:  # pragma: no cover - guarded by regex
        raise ValueError(f"unparsable date: {date_s!r}")

    candidates = list(_TIME_FORMATS)
    if not ampm:
        candidates = [f for f in candidates if "%p" not in f]
    else:
        candidates = [f for f in candidates if "%p" in f]
        time_s = f"{time_s} {ampm.replace('.', '').replace(' ', '').upper()}"

    for fmt in candidates:
        try:
            t = datetime.strptime(time_s, fmt)
            return date_part.replace(
                hour=t.hour, minute=t.minute, second=t.second
            )
        except ValueError:
            continue
    raise ValueError(f"unparsable time: {time_s!r}")


def _classify_media(kind: str, name: str | None) -> tuple[MsgType, str | None]:
    k = kind.lower()
    if name:
        ext = Path(name).suffix.lower()
        if k == "ptt" or "PTT-" in name.upper() or ext in _VOICE_EXT or ".opus" in name.lower():
            return "voice", name
        if ext in _IMAGE_EXT or name.upper().startswith(("IMG-", "PHOTO-")):
            return "image", name
        if ext in _VIDEO_EXT or name.upper().startswith("VID-"):
            return "video", name
        if name.upper().startswith("GIF-"):
            return "gif", name
        if name.upper().startswith("STK-"):
            return "sticker", name
        if ext in _DOC_EXT:
            return "document", name
        return "media", name

    if k in {"media", "media omesso", "media omessi", "allegato", "allegato omesso", "allegati omessi"}:
        return "media", None
    if k in {"image", "gif", "sticker", "contact", "posizione"}:
        return ("location" if k == "posizione" else k), None
    if k in {"video", "audio", "document"}:
        return k, None
    return "media", None


def _split_sender(rest: str) -> tuple[str | None, str]:
    """Return (sender, text). ``sender`` is None for messages without one."""
    sep = rest.find(": ")
    if sep == -1:
        return None, rest
    sender = rest[:sep].strip()
    text = rest[sep + 2 :]
    # A sender is a short name; long prefixes almost always mean a system line
    # whose text happened to contain ": ".
    if not sender or len(sender) > 60:
        return None, rest
    return sender, text


def parse_line_timestamp(line: str, tz: ZoneInfo) -> tuple[int, str, str, str, str] | None:
    m = _LINE_RE.match(line)
    if not m:
        return None
    dt = _parse_datetime(m.group("date"), m.group("time"), m.group("ampm"))
    local = dt.replace(tzinfo=tz)
    return (
        int(local.timestamp()),
        dt.strftime("%Y-%m-%d %H:%M:%S"),
        dt.strftime("%Y-%m-%d"),
        dt.strftime("%H:%M"),
        m.group("rest"),
    )


def parse_stream(lines: Iterator[str], tz_name: str = "Europe/Rome") -> Iterator[tuple[ParsedMessage, ParseStats]]:
    """Yield parsed messages one by one, folding continuation lines in.

    Yields ``(message, running_stats)`` so callers can stream. The same mutable
    ``ParseStats`` object is reused between yields.
    """
    tz = ZoneInfo(tz_name)
    stats = ParseStats()
    current: ParsedMessage | None = None

    for idx, raw in enumerate(lines, start=1):
        stats.lines += 1
        line = raw.rstrip("\n").rstrip("\r")
        parsed = parse_line_timestamp(line, tz)

        if parsed is None:
            if current is None:
                # Stray content before any timestamped line: keep as raw system.
                current = ParsedMessage(
                    ts=0,
                    ts_local="",
                    local_date="",
                    local_time="",
                    sender=None,
                    text=line,
                    raw_line=line,
                    msg_type="system",
                    line_no=idx,
                )
                continue
            current.text += "\n" + line
            current.raw_line += "\n" + line
            stats.continuations += 1
            continue

        if current is not None:
            yield current, stats

        ts, ts_local, local_date, local_time, rest = parsed
        current = _build_message(ts, ts_local, local_date, local_time, rest, line, idx)
        stats.messages += 1
        stats.bump(current.msg_type)

    if current is not None:
        yield current, stats


def _build_message(
    ts: int,
    ts_local: str,
    local_date: str,
    local_time: str,
    rest: str,
    raw_line: str,
    line_no: int,
) -> ParsedMessage:
    media_ref: str | None = None

    if _CALL_RE.match(rest):
        return ParsedMessage(ts, ts_local, local_date, local_time, None, rest, raw_line, "call", None, line_no)

    if _SYSTEM_RE.search(rest) and ": " not in rest:
        return ParsedMessage(ts, ts_local, local_date, local_time, None, rest, raw_line, "system", None, line_no)

    sender, text = _split_sender(rest)

    if sender is None:
        return ParsedMessage(ts, ts_local, local_date, local_time, None, text, raw_line, "system", None, line_no)

    # Media placeholders.
    m = _MEDIA_OMITTED_RE.match(text.strip())
    if m:
        kind, ref = _classify_media(m.group("kind"), None)
        return ParsedMessage(ts, ts_local, local_date, local_time, sender, text, raw_line, kind, ref, line_no)

    m = _FILE_ATTACHED_RE.search(text)
    if m:
        name = m.group("name")
        kind, ref = _classify_media("", name)
        media_ref = ref
        return ParsedMessage(ts, ts_local, local_date, local_time, sender, text, raw_line, kind, media_ref, line_no)

    if _DELETED_RE.search(text):
        return ParsedMessage(ts, ts_local, local_date, local_time, sender, text, raw_line, "deleted", None, line_no)

    m = _EDITED_RE.search(text)
    if m:
        cleaned = text[: m.start()].rstrip()
        return ParsedMessage(ts, ts_local, local_date, local_time, sender, cleaned, raw_line, "edited", None, line_no)

    return ParsedMessage(ts, ts_local, local_date, local_time, sender, text, raw_line, "text", None, line_no)


def parse_file(path: Path, tz_name: str = "Europe/Rome") -> Iterator[tuple[ParsedMessage, ParseStats]]:
    with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
        yield from parse_stream((line for line in fh), tz_name=tz_name)


def chat_id_from_filename(path: Path) -> str:
    stem = path.stem
    for prefix in ("WhatsApp Chat with ", "Chat WhatsApp con ", "WhatsApp Chat - ", "Chat de WhatsApp con "):
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
    return re.sub(r"\s+", " ", stem).strip() or path.stem
