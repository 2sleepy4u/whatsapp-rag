from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from chat_rag.db import stats as stats_mod
from chat_rag.db.db import open_db

CHAT = "c"
A = "c::Alice"
B = "c::Bob"
BASE = datetime(2023, 11, 14, 23, 0, 0)


def _insert(conn, seconds: int, sender: str, text: str, msg_type: str = "text") -> None:
    local = BASE + timedelta(seconds=seconds)
    conn.execute(
        """
        INSERT INTO messages
            (id, chat_id, sender_id, ts, ts_local, local_date, local_time, text,
             raw_line, msg_type, source_file, line_no, ingested_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            f"m{seconds}", CHAT, sender, int(local.timestamp()),
            local.strftime("%Y-%m-%d %H:%M:%S"), local.strftime("%Y-%m-%d"),
            local.strftime("%H:%M"), text, text, msg_type, "t", 1, 0,
        ),
    )


@pytest.fixture()
def conn(tmp_path):
    c = open_db(tmp_path / "chat.db")
    c.execute("INSERT INTO chats (chat_id, name) VALUES (?, ?)", (CHAT, CHAT))
    for sid, name in ((A, "Alice"), (B, "Bob")):
        c.execute("INSERT INTO senders (sender_id, chat_id, name) VALUES (?, ?, ?)", (sid, CHAT, name))
    _insert(c, 0, A, "ciao bello")
    _insert(c, 60, B, "ehi")
    _insert(c, 120, A, "😀😀")
    _insert(c, 100000, B, "ciao")
    _insert(c, 100060, A, "bene")
    c.execute("INSERT INTO ingestion_runs (source_file, file_hash, chat_id, started_at) VALUES ('t','h',?,0)", (CHAT,))
    c.commit()
    return c


def test_per_sender(conn):
    rows = stats_mod.per_sender(conn, CHAT)
    by_name = {s.name: s for s in rows}
    assert by_name["Alice"].messages == 3
    assert by_name["Bob"].messages == 2
    assert round(by_name["Alice"].share, 2) == 0.6


def test_response_times(conn):
    rows = {r.variant: r for r in stats_mod.response_times(conn, CHAT, session_gap_seconds=3600)}
    assert rows["any"].count == 4
    assert rows["any"].median_sec == 60
    assert rows["reply"].count == 4
    assert rows["reply-session"].count == 3


def test_word_frequency(conn):
    words = dict(stats_mod.word_frequency(conn, CHAT, stopwords=True))
    assert words["ciao"] == 2
    assert words["bello"] == 1


def test_emoji_frequency(conn):
    emojis = dict(stats_mod.emoji_frequency(conn, CHAT))
    assert emojis["😀"] == 2


def test_sessions(conn):
    sessions = stats_mod.sessions(conn, CHAT, gap_seconds=3600)
    assert len(sessions) == 2
    assert sessions[0].messages == 3
    assert sessions[0].starter == "Alice"
    assert sessions[1].starter == "Bob"
    assert sessions[1].gap_before_sec == 100000 - 120


def test_volume(conn):
    rows = stats_mod.volume(conn, CHAT, bucket="month")
    assert rows == [("2023-11", 5)]


def test_time_of_day(conn):
    data = stats_mod.time_of_day(conn, CHAT)
    hours = dict(data["hours"])
    assert hours[23] == 3
    assert hours[2] == 2
