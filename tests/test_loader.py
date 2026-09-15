from __future__ import annotations

from pathlib import Path

from chat_rag.db.db import open_db
from chat_rag.ingest.loader import ingest_file

FIXTURE = Path(__file__).parent / "fixtures" / "android_chat.txt"


def test_ingest_and_dedup(tmp_path):
    conn = open_db(tmp_path / "chat.db")
    result = ingest_file(conn, FIXTURE, chat_id="test", tz="Europe/Rome")
    assert result.inserted == 11
    assert result.parsed == 11
    assert result.by_type["text"] == 5
    assert result.by_type["system"] == 1
    assert result.by_type["voice"] == 1
    assert result.by_type["media"] == 1
    assert result.by_type["deleted"] == 1
    assert result.by_type["edited"] == 1
    assert result.by_type["call"] == 1

    chat = conn.execute("SELECT * FROM chats WHERE chat_id='test'").fetchone()
    assert chat["message_count"] == 11
    assert chat["is_group"] == 0

    # Re-ingesting the same file is skipped by hash.
    again = ingest_file(conn, FIXTURE, chat_id="test", tz="Europe/Rome")
    assert again.already_ingested
    assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 11

    # Forced re-ingest keeps stable ids and does not duplicate.
    forced = ingest_file(conn, FIXTURE, chat_id="test", tz="Europe/Rome", force=True)
    assert forced.inserted == 0
    assert forced.skipped == 11
    assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 11


def test_fts_search(tmp_path):
    conn = open_db(tmp_path / "chat.db")
    ingest_file(conn, FIXTURE, chat_id="test", tz="Europe/Rome")
    rows = conn.execute(
        "SELECT m.text FROM messages_fts f JOIN messages m ON m.rowid = f.rowid "
        "WHERE messages_fts MATCH 'bene' ORDER BY rank"
    ).fetchall()
    assert any("Tutto bene" in r["text"] for r in rows)
