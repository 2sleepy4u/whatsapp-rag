from __future__ import annotations

import pytest

from chat_rag.db.db import open_db
from chat_rag.embed.index import chroma_client, ensure_collection, index_messages
from chat_rag.rag.tools import ToolContext

CHAT = "c"
A = "c::Alice"
B = "c::Bob"

MESSAGES = [
    (1, A, "andiamo al mare domenica", "text"),
    (2, B, "volentieri porto la bici", "text"),
    (3, A, "mi raccomando non dimenticare", "text"),
    (4, B, "ok ci vediamo al solito posto", "text"),
    (5, A, "che bello il mare", "text"),
]


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder (similar text -> similar vector)."""

    model = "fake-embed"

    def __init__(self, dim: int = 32) -> None:
        self.dim = dim

    def embed(self, texts):
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            for token in text.lower().split():
                vec[hash(token) % self.dim] += 1.0
            norm = sum(x * x for x in vec) ** 0.5 or 1.0
            out.append([x / norm for x in vec])
        return out


def seed_db(conn) -> None:
    conn.execute("INSERT INTO chats (chat_id, name) VALUES (?, ?)", (CHAT, CHAT))
    for sid, name in ((A, "Alice"), (B, "Bob")):
        conn.execute("INSERT INTO senders (sender_id, chat_id, name) VALUES (?, ?, ?)", (sid, CHAT, name))
    for i, sid, text, mtype in MESSAGES:
        conn.execute(
            """
            INSERT INTO messages
                (id, chat_id, sender_id, ts, ts_local, local_date, local_time, text,
                 raw_line, msg_type, source_file, line_no, ingested_at)
            VALUES (?, ?, ?, ?, ?, '2024-03-01', '10:00', ?, ?, ?, 't', ?, 0)
            """,
            (f"{i:016x}", CHAT, sid, 1700000000 + i * 60, f"2024-03-01 10:0{i}:00", text, text, mtype, i),
        )
    conn.commit()


@pytest.fixture()
def ctx(tmp_path):
    conn = open_db(tmp_path / "chat.db")
    seed_db(conn)
    client = chroma_client(tmp_path / "chroma")
    col = ensure_collection(client, "messages")
    embedder = FakeEmbedder()
    index_messages(conn, col, embedder)
    return ToolContext(conn=conn, collection=col, embedder=embedder)


@pytest.fixture()
def raw_conn(tmp_path):
    conn = open_db(tmp_path / "chat.db")
    seed_db(conn)
    return conn
