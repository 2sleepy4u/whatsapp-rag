from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .config import load_settings
from .db import stats as stats_mod
from .db.db import open_db
from .ingest.loader import ingest_file
from .ingest.parser_android import ParseStats, parse_file

app = typer.Typer(help="Local RAG + analytics over exported WhatsApp chats.", no_args_is_help=True)
stats_app = typer.Typer(help="Statistics over ingested chats.", no_args_is_help=True)
app.add_typer(stats_app, name="stats")
console = Console()


@app.command("import")
def import_chats(
    paths: list[Path] = typer.Argument(..., exists=True, readable=True, help="WhatsApp .txt export(s)"),
    chat_id: str | None = typer.Option(None, "--chat-id", help="Override the chat id (default: filename)"),
    tz: str | None = typer.Option(None, "--tz", help="Timezone for timestamps"),
    force: bool = typer.Option(False, "--force", help="Re-parse even if this exact file was ingested"),
) -> None:
    """Parse and load one or more Android WhatsApp exports into SQLite."""
    settings = load_settings()
    settings.ensure_dirs()
    tz = tz or settings.tz
    conn = open_db(settings.db_path)

    for path in paths:
        with console.status(f"Parsing {path.name}..."):
            result = ingest_file(conn, path, chat_id=chat_id, tz=tz, force=force)
        if result.already_ingested:
            console.print(f"[yellow]Skipped[/] {path.name}: already ingested (use --force to reimport)")
            continue
        console.print(f"[green]Done[/] {result.summary()}")


@app.command()
def preview(
    path: Path = typer.Argument(..., exists=True, readable=True),
    tz: str | None = typer.Option(None, "--tz"),
    samples: int = typer.Option(8, "--samples", help="Sample lines per message type"),
) -> None:
    """Dry-run the parser on a file: report types/senders without touching the DB."""
    settings = load_settings()
    tz = tz or settings.tz
    by_type: dict[str, list] = {}
    senders: dict[str, int] = {}
    first_ts = last_ts = None

    with console.status(f"Scanning {path.name}..."):
        final = None
        for msg, stats in parse_file(path, tz_name=tz):
            final = stats
            by_type.setdefault(msg.msg_type, []).append(msg)
            if msg.sender:
                senders[msg.sender] = senders.get(msg.sender, 0) + 1
            if msg.ts:
                first_ts = msg.ts if first_ts is None else min(first_ts, msg.ts)
                last_ts = msg.ts if last_ts is None else max(last_ts, msg.ts)
        stats = final or ParseStats()

    console.print(
        f"[bold]{path.name}[/]  lines={stats.lines:,} messages={stats.messages:,} "
        f"continuations={stats.continuations:,}"
    )
    console.print(f"range: {_fmt_ts(first_ts)} -> {_fmt_ts(last_ts)}")

    table = Table(title="Types")
    table.add_column("type")
    table.add_column("count", justify="right")
    for t, msgs in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        table.add_row(t, f"{len(msgs):,}")
    console.print(table)

    stable = Table(title=f"Senders ({len(senders)})")
    stable.add_column("name")
    stable.add_column("messages", justify="right")
    for name, n in sorted(senders.items(), key=lambda kv: -kv[1]):
        stable.add_row(name, f"{n:,}")
    console.print(stable)

    for t, msgs in sorted(by_type.items()):
        console.print(f"\n[bold]{t}[/] samples:")
        for msg in msgs[:samples]:
            who = msg.sender or "(system)"
            text = msg.text.replace("\n", "⏎")[:100]
            console.print(f"  {msg.ts_local}  [cyan]{who}[/]: {text}")


@app.command()
def info() -> None:
    """Show ingested chats, senders and message type breakdown."""
    settings = load_settings()
    if not settings.db_path.exists():
        console.print("[red]No database yet. Run `chat-rag import <file.txt>` first.[/]")
        raise typer.Exit(code=1)

    conn = open_db(settings.db_path)

    chats = conn.execute(
        "SELECT chat_id, name, is_group, message_count, first_ts, last_ts FROM chats ORDER BY message_count DESC"
    ).fetchall()

    table = Table(title="Chats")
    table.add_column("chat_id")
    table.add_column("group")
    table.add_column("messages", justify="right")
    table.add_column("from")
    table.add_column("to")
    for c in chats:
        table.add_row(
            c["chat_id"],
            "yes" if c["is_group"] else "no",
            f"{c['message_count']:,}",
            _fmt_ts(c["first_ts"]),
            _fmt_ts(c["last_ts"]),
        )
    console.print(table)

    types = conn.execute(
        "SELECT msg_type, COUNT(*) n FROM messages GROUP BY msg_type ORDER BY n DESC"
    ).fetchall()
    ttable = Table(title="Message types")
    ttable.add_column("type")
    ttable.add_column("count", justify="right")
    for t in types:
        ttable.add_row(t["msg_type"], f"{t['n']:,}")
    console.print(ttable)

    senders = conn.execute(
        "SELECT name, message_count, chat_id FROM senders ORDER BY message_count DESC LIMIT 25"
    ).fetchall()
    stable = Table(title="Top senders")
    stable.add_column("name")
    stable.add_column("chat")
    stable.add_column("messages", justify="right")
    for s in senders:
        stable.add_row(s["name"], s["chat_id"], f"{s['message_count']:,}")
    console.print(stable)
    conn.close()


def _fmt_ts(ts: int | None) -> str:
    if not ts:
        return "-"
    from datetime import datetime

    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")


def _open_stats_db():
    settings = load_settings()
    if not settings.db_path.exists():
        console.print("[red]No database yet. Run `chat-rag import <file.txt>` first.[/]")
        raise typer.Exit(code=1)
    return open_db(settings.db_path)


def _bar(value: int, max_value: int, width: int = 24) -> str:
    if max_value <= 0:
        return ""
    filled = round(value / max_value * width)
    return "█" * filled


@stats_app.command("per-sender")
def stats_per_sender(
    chat: str | None = typer.Option(None, "--chat", help="Filter by chat_id"),
    date_from: str | None = typer.Option(None, "--from", help="YYYY-MM-DD"),
    date_to: str | None = typer.Option(None, "--to", help="YYYY-MM-DD"),
) -> None:
    """Messages, share and length per sender."""
    conn = _open_stats_db()
    rows = stats_mod.per_sender(conn, chat, date_from, date_to)
    table = Table(title="Per sender")
    for col, just in [
        ("name", "left"), ("messages", "right"), ("share", "right"),
        ("avg chars", "right"), ("words", "right"), ("from", "left"), ("to", "left"),
    ]:
        table.add_column(col, justify=just)
    for s in rows:
        table.add_row(
            s.name, f"{s.messages:,}", f"{s.share * 100:.1f}%",
            f"{s.avg_chars:.1f}", f"{s.words:,}", s.first_local, s.last_local,
        )
    console.print(table)
    conn.close()


@stats_app.command("response-time")
def stats_response_time(
    chat: str | None = typer.Option(None, "--chat"),
    date_from: str | None = typer.Option(None, "--from", help="YYYY-MM-DD"),
    date_to: str | None = typer.Option(None, "--to", help="YYYY-MM-DD"),
    session_gap: int = typer.Option(3600, "--session-gap", help="Seconds that split a conversation session"),
) -> None:
    """Response-time distribution (any gap / reply turns / replies in-session)."""
    conn = _open_stats_db()
    rows = stats_mod.response_times(conn, chat, date_from, date_to, session_gap_seconds=session_gap)
    table = Table(title="Response time")
    for col in ["variant", "count", "mean", "median", "p90", "min", "max"]:
        table.add_column(col, justify="right" if col != "variant" else "left")
    for r in rows:
        table.add_row(
            r.variant, f"{r.count:,}", r.human(r.mean_sec), r.human(r.median_sec),
            r.human(r.p90_sec), r.human(r.min_sec), r.human(r.max_sec),
        )
    console.print(table)
    console.print(
        "[dim]any = every consecutive pair · reply = sender change · "
        "reply-session = sender change within the session gap[/]"
    )
    conn.close()


@stats_app.command("words")
def stats_words(
    chat: str | None = typer.Option(None, "--chat"),
    date_from: str | None = typer.Option(None, "--from", help="YYYY-MM-DD"),
    date_to: str | None = typer.Option(None, "--to", help="YYYY-MM-DD"),
    top: int = typer.Option(40, "--top"),
    min_len: int = typer.Option(3, "--min-len"),
    stopwords: bool = typer.Option(True, "--stopwords/--no-stopwords"),
) -> None:
    """Most frequent words."""
    conn = _open_stats_db()
    rows = stats_mod.word_frequency(conn, chat, date_from, date_to, top, min_len, stopwords)
    max_n = rows[0][1] if rows else 0
    table = Table(title="Word frequency")
    table.add_column("word")
    table.add_column("count", justify="right")
    table.add_column("", justify="left")
    for word, n in rows:
        table.add_row(word, f"{n:,}", _bar(n, max_n))
    console.print(table)
    conn.close()


@stats_app.command("emojis")
def stats_emojis(
    chat: str | None = typer.Option(None, "--chat"),
    date_from: str | None = typer.Option(None, "--from", help="YYYY-MM-DD"),
    date_to: str | None = typer.Option(None, "--to", help="YYYY-MM-DD"),
    top: int = typer.Option(30, "--top"),
) -> None:
    """Most frequent emojis."""
    conn = _open_stats_db()
    rows = stats_mod.emoji_frequency(conn, chat, date_from, date_to, top)
    max_n = rows[0][1] if rows else 0
    table = Table(title="Emoji frequency")
    table.add_column("emoji")
    table.add_column("count", justify="right")
    table.add_column("", justify="left")
    for e, n in rows:
        table.add_row(e, f"{n:,}", _bar(n, max_n))
    console.print(table)
    conn.close()


@stats_app.command("sessions")
def stats_sessions(
    chat: str | None = typer.Option(None, "--chat"),
    date_from: str | None = typer.Option(None, "--from", help="YYYY-MM-DD"),
    date_to: str | None = typer.Option(None, "--to", help="YYYY-MM-DD"),
    gap: int = typer.Option(3600, "--gap", help="Seconds of silence that start a new session"),
    top: int = typer.Option(15, "--top", help="How many sessions/gaps to list"),
) -> None:
    """Conversation sessions, who starts them, and the longest silences."""
    conn = _open_stats_db()
    sessions = stats_mod.sessions(conn, chat, date_from, date_to, gap_seconds=gap)
    console.print(f"[bold]{len(sessions):,}[/] sessions (gap > {gap}s)")

    starters: dict[str, int] = {}
    for s in sessions:
        starters[s.starter] = starters.get(s.starter, 0) + 1
    table = Table(title="Who starts conversations")
    table.add_column("name")
    table.add_column("sessions", justify="right")
    table.add_column("share", justify="right")
    for name, n in sorted(starters.items(), key=lambda kv: -kv[1]):
        table.add_row(name, f"{n:,}", f"{n / len(sessions) * 100:.1f}%" if sessions else "-")
    console.print(table)

    longest = sorted(sessions, key=lambda s: s.messages, reverse=True)[:top]
    ltable = Table(title="Longest sessions")
    for col in ["start", "end", "messages", "starter"]:
        ltable.add_column(col, justify="right" if col == "messages" else "left")
    for s in longest:
        ltable.add_row(s.start_local, s.end_local, f"{s.messages:,}", s.starter)
    console.print(ltable)

    gaps = [s for s in sessions if s.gap_before_sec is not None]
    biggest = sorted(gaps, key=lambda s: s.gap_before_sec or 0, reverse=True)[:top]
    gtable = Table(title="Longest silences")
    gtable.add_column("resumed at")
    gtable.add_column("silence", justify="right")
    gtable.add_column("starter", justify="left")
    for s in biggest:
        gtable.add_row(s.start_local, stats_mod._human_seconds(float(s.gap_before_sec or 0)), s.starter)
    console.print(gtable)
    conn.close()


@stats_app.command("volume")
def stats_volume(
    chat: str | None = typer.Option(None, "--chat"),
    date_from: str | None = typer.Option(None, "--from", help="YYYY-MM-DD"),
    date_to: str | None = typer.Option(None, "--to", help="YYYY-MM-DD"),
    bucket: str = typer.Option("month", "--bucket", help="day|week|month|year"),
) -> None:
    """Message volume over time."""
    if bucket not in {"day", "week", "month", "year"}:
        raise typer.BadParameter("bucket must be day|week|month|year")
    conn = _open_stats_db()
    rows = stats_mod.volume(conn, chat, date_from, date_to, bucket)
    max_n = max((n for _, n in rows), default=0)
    table = Table(title=f"Volume by {bucket}")
    table.add_column("period")
    table.add_column("messages", justify="right")
    table.add_column("", justify="left")
    for period, n in rows:
        table.add_row(period, f"{n:,}", _bar(n, max_n))
    console.print(table)
    conn.close()


_WEEKDAYS = ["Dom", "Lun", "Mar", "Mer", "Gio", "Ven", "Sab"]


@stats_app.command("time-of-day")
def stats_time_of_day(
    chat: str | None = typer.Option(None, "--chat"),
    date_from: str | None = typer.Option(None, "--from", help="YYYY-MM-DD"),
    date_to: str | None = typer.Option(None, "--to", help="YYYY-MM-DD"),
) -> None:
    """Activity by hour and weekday."""
    conn = _open_stats_db()
    data = stats_mod.time_of_day(conn, chat, date_from, date_to)

    hmax = max((n for _, n in data["hours"]), default=0)
    htable = Table(title="By hour")
    htable.add_column("hour")
    htable.add_column("messages", justify="right")
    htable.add_column("", justify="left")
    for h, n in data["hours"]:
        htable.add_row(f"{h:02d}", f"{n:,}", _bar(n, hmax))
    console.print(htable)

    wmax = max((n for _, n in data["weekdays"]), default=0)
    wtable = Table(title="By weekday")
    wtable.add_column("day")
    wtable.add_column("messages", justify="right")
    wtable.add_column("", justify="left")
    for d, n in data["weekdays"]:
        wtable.add_row(_WEEKDAYS[d] if 0 <= d < 7 else str(d), f"{n:,}", _bar(n, wmax))
    console.print(wtable)
    conn.close()


if __name__ == "__main__":
    app()
