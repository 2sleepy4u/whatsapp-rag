from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .config import load_settings
from .db.db import open_db
from .ingest.loader import ingest_file
from .ingest.parser_android import ParseStats, parse_file

app = typer.Typer(help="Local RAG + analytics over exported WhatsApp chats.", no_args_is_help=True)
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


if __name__ == "__main__":
    app()
