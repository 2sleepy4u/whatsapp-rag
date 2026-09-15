from __future__ import annotations

from chat_rag.db.db import open_db
from chat_rag.embed.index import (
    build_windows,
    chroma_client,
    ensure_collection,
    index_messages,
    index_windows,
    reset_index,
)

CHAT = "c"
A = "c::Alice"
B = "c::Bob"


class FakeEmbedder:
    model = "fake-embed"

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            for token in text.lower().split():
                vec[hash(token) % self.dim] += 1.0
            norm = sum(x * x for x in vec) ** 0.5 or 1.0
            out.append([x / norm for x in vec])
        return out


def _seed(conn):
    conn.execute("INSERT INTO chats (chat_id, name) VALUES (?, ?)", (CHAT, CHAT))
    for sid, name in ((A, "Alice"), (B, "Bob")):
        conn.execute("INSERT INTO senders (sender_id, chat_id, name) VALUES (?, ?, ?)", (sid, CHAT, name))
    rows = [
        (1, A, "ciao come va"),
        (2, B, "tutto bene grazie"),
        (3, A, "andiamo al mare?"),
        (4, B, "volentieri domenica"),
        (5, A, "perfetto allora"),
    ]
    for i, sid, text in rows:
        conn.execute(
            """
            INSERT INTO messages
                (id, chat_id, sender_id, ts, ts_local, local_date, local_time, text,
                 raw_line, msg_type, source_file, line_no, ingested_at)
            VALUES (?, ?, ?, ?, ?, '2024-01-01', '10:00', ?, ?, 'text', 't', ?, 0)
            """,
            (f"m{i}", CHAT, sid, 1700000000 + i * 60, f"2024-01-01 10:0{i}:00", text, text, i),
        )
    # A system message that must not be indexed.
    conn.execute(
        """
        INSERT INTO messages
            (id, chat_id, sender_id, ts, ts_local, local_date, local_time, text,
             raw_line, msg_type, source_file, line_no, ingested_at)
        VALUES ('sys', ?, NULL, 1700000300, '2024-01-01 10:05:00', '2024-01-01', '10:05',
                'Luca ha creato il gruppo', 'Luca ha creato il gruppo', 'system', 't', 0, 0)
        """,
        (CHAT,),
    )
    conn.commit()


def test_index_messages_and_incremental(tmp_path):
    conn = open_db(tmp_path / "chat.db")
    _seed(conn)
    client = chroma_client(tmp_path / "chroma")
    col = ensure_collection(client, "messages")
    embedder = FakeEmbedder()

    stats = index_messages(conn, col, embedder)
    assert stats.total == 5
    assert stats.embedded == 5
    assert stats.dim == 16
    assert col.count() == 5

    again = index_messages(conn, col, embedder)
    assert again.pending == 0
    assert again.embedded == 0
    assert again.skipped == 5
    assert col.count() == 5


def test_index_windows(tmp_path):
    conn = open_db(tmp_path / "chat.db")
    _seed(conn)
    client = chroma_client(tmp_path / "chroma")
    col = ensure_collection(client, "messages")
    embedder = FakeEmbedder()

    stats = index_windows(conn, col, embedder, size=2, stride=1)
    # 5 messages, window 2, stride 1 -> windows [0,1][1,2][2,3][3,4]
    assert stats.total == 4
    assert stats.embedded == 4
    assert stats.kind == "window"


def test_recreate_resets(tmp_path):
    conn = open_db(tmp_path / "chat.db")
    _seed(conn)
    client = chroma_client(tmp_path / "chroma")
    col = ensure_collection(client, "messages")
    embedder = FakeEmbedder()
    index_messages(conn, col, embedder)

    reset_index(conn, client, "messages", embedder.model)
    col = ensure_collection(client, "messages")
    assert col.count() == 0
    assert conn.execute("SELECT COUNT(*) FROM embedding_index").fetchone()[0] == 0

    stats = index_messages(conn, col, embedder)
    assert stats.embedded == 5


def test_semantic_query(tmp_path):
    conn = open_db(tmp_path / "chat.db")
    _seed(conn)
    client = chroma_client(tmp_path / "chroma")
    col = ensure_collection(client, "messages")
    embedder = FakeEmbedder()
    index_messages(conn, col, embedder)

    qvec = embedder.embed(["mare domenica"])[0]
    res = col.query(query_embeddings=[qvec], n_results=2, where={"kind": "message"})
    assert res["ids"][0]
    docs = res["documents"][0]
    assert any("mare" in d or "domenica" in d for d in docs)


def test_build_windows_tail_included():
    class Row(dict):
        def __getitem__(self, k):
            return dict.__getitem__(self, k)

    rows = []
    for i in range(5):
        rows.append(Row({"id": f"m{i}", "ts": i, "local_date": "2024-01-01", "chat_id": "c",
                         "text": f"t{i}", "sender_name": "A", "sender_id": "c::A", "msg_type": "text"}))
    windows = build_windows(rows, size=2, stride=1)
    assert len(windows) == 4
    assert windows[-1].meta["end_id"] == "m4"
