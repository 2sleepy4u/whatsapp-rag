from __future__ import annotations

import pytest

from chat_rag.analytics import inside_jokes
from chat_rag.db.db import open_db

CHAT = "c"
A = "c::Luca"
B = "c::Giulia"

ROWS = [
    ("2024-01-05", A, "oggi parliamo della papera"),
    ("2024-01-06", B, "ma che bella giornata"),
    ("2024-02-10", A, "la papera è tornata"),
    ("2024-02-11", B, "la papera vola alto"),
    ("2024-03-01", A, "la papera domina il mondo"),
    ("2024-03-02", A, "niente da fare oggi"),
]


@pytest.fixture()
def conn(tmp_path):
    c = open_db(tmp_path / "chat.db")
    c.execute("INSERT INTO chats (chat_id, name) VALUES (?, ?)", (CHAT, CHAT))
    for sid, name in ((A, "Luca"), (B, "Giulia")):
        c.execute("INSERT INTO senders (sender_id, chat_id, name) VALUES (?, ?, ?)", (sid, CHAT, name))
    for i, (day, sid, text) in enumerate(ROWS):
        c.execute(
            """
            INSERT INTO messages
                (id, chat_id, sender_id, ts, ts_local, local_date, local_time, text,
                 raw_line, msg_type, source_file, line_no, ingested_at)
            VALUES (?, ?, ?, ?, ?, ?, '10:00', ?, ?, 'text', 't', ?, 0)
            """,
            (f"{i:016x}", CHAT, sid, 1700000000 + i * 3600, f"{day} 10:00:00", day, text, text, i),
        )
    c.commit()
    return c


def test_candidate_ngrams_finds_phrase(conn):
    candidates = inside_jokes.candidate_ngrams(conn, min_count=3)
    phrases = {c.phrase: c for c in candidates}
    assert "la papera" in phrases
    assert phrases["la papera"].count == 3
    assert phrases["la papera"].first == "2024-02-10"
    assert phrases["la papera"].last == "2024-03-01"
    assert ("Luca", 2) in phrases["la papera"].senders


def test_candidate_respects_min_count(conn):
    assert inside_jokes.candidate_ngrams(conn, min_count=10) == []


def test_phrase_timeline(conn):
    series = dict(inside_jokes.phrase_timeline(conn, "la papera", bucket="month"))
    assert series == {"2024-02": 2, "2024-03": 1}


def test_phrase_evidence(conn):
    evidence = inside_jokes.phrase_evidence(conn, "la papera", top=10)
    assert len(evidence) == 3
    assert all(e["id"] for e in evidence)
    assert all("papera" in e["text"] for e in evidence)


def test_overlap_suppression(tmp_path):
    c = open_db(tmp_path / "chat.db")
    c.execute("INSERT INTO chats (chat_id, name) VALUES ('x', 'x')")
    c.execute("INSERT INTO senders (sender_id, chat_id, name) VALUES ('x::A', 'x', 'A')")
    for i in range(6):
        text = "oggi ciao bello come stai"
        c.execute(
            """
            INSERT INTO messages
                (id, chat_id, sender_id, ts, ts_local, local_date, local_time, text,
                 raw_line, msg_type, source_file, line_no, ingested_at)
            VALUES (?, 'x', 'x::A', ?, ?, '2024-01-01', '10:00', ?, ?, 'text', 't', ?, 0)
            """,
            (f"{i:016x}", 1700000000 + i, f"2024-01-01 10:0{i}:00", text, text, i),
        )
    c.commit()
    phrases = [p.phrase for p in inside_jokes.candidate_ngrams(c, min_count=5, max_n=4)]
    assert "ciao bello come stai" in phrases
    assert "ciao bello" not in phrases
    assert "bello come stai" not in phrases

