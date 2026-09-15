from __future__ import annotations

import json
from dataclasses import dataclass

from fastapi.testclient import TestClient

from chat_rag.config import Settings
from chat_rag.db.db import open_db
from chat_rag.rag.citations import Citation
from chat_rag.rag.service import Answer
from chat_rag.server.app import create_app

MSG_ID = f"{1:016x}"


def _settings(tmp_path) -> Settings:
    return Settings(
        data_dir=tmp_path,
        tz="Europe/Rome",
        ollama_host="http://localhost:11434",
        llm_model="test",
        embed_model="test-embed",
        llm_num_predict=64,
        llm_think=False,
    )


def _seed(tmp_path) -> Settings:
    settings = _settings(tmp_path)
    settings.ensure_dirs()
    conn = open_db(settings.db_path)
    conn.execute("INSERT INTO chats (chat_id, name) VALUES ('c', 'c')")
    conn.execute("INSERT INTO senders (sender_id, chat_id, name) VALUES ('c::Alice', 'c', 'Alice')")
    conn.execute(
        """
        INSERT INTO messages
            (id, chat_id, sender_id, ts, ts_local, local_date, local_time, text,
             raw_line, msg_type, source_file, line_no, ingested_at)
        VALUES (?, 'c', 'c::Alice', 1700000000, '2024-03-01 10:00:00', '2024-03-01', '10:00',
                'andiamo al mare domenica', 'andiamo al mare domenica', 'text', 't', 0, 0)
        """,
        (MSG_ID,),
    )
    conn.commit()
    conn.close()
    return settings


@dataclass
class FakeAsk:
    calls: list = None

    def __post_init__(self) -> None:
        self.calls = []

    def __call__(self, question, chat, max_steps, on_event, on_token):
        self.calls.append({"question": question, "chat": chat, "max_steps": max_steps})
        if on_event:
            on_event("tool_call", {"name": "semantic_search", "arguments": {"query": question}})
        for token in ["Ciao ", "mondo"]:
            if on_token:
                on_token(token)
        if on_event:
            on_event("tool_result", {"name": "semantic_search", "count": 1})
        return Answer(
            question=question,
            answer=f"Ciao mondo [{MSG_ID}]",
            citations=[Citation(MSG_ID, "2024-03-01", "10:00", "Alice", "andiamo al mare domenica")],
            steps=[{"tool": "semantic_search", "arguments": {"query": question}}],
        )


def _client(tmp_path):
    settings = _seed(tmp_path)
    fake = FakeAsk()
    return TestClient(create_app(settings, ask=fake)), fake


def test_index_page(tmp_path):
    client, _ = _client(tmp_path)
    r = client.get("/")
    assert r.status_code == 200
    assert "chat-rag" in r.text


def test_chats(tmp_path):
    client, _ = _client(tmp_path)
    data = client.get("/api/chats").json()
    assert data["chats"][0]["chat_id"] == "c"
    assert data["chats"][0]["messages"] == 1


def test_message_and_context(tmp_path):
    client, _ = _client(tmp_path)
    data = client.get(f"/api/messages/{MSG_ID}").json()
    assert data["sender"] == "Alice" and "mare" in data["text"]

    ctx = client.get(f"/api/messages/{MSG_ID}?context=2").json()
    assert any(m["id"] == MSG_ID for m in ctx["messages"])

    assert client.get("/api/messages/deadbeefdeadbeef").status_code == 404


def test_ask_json(tmp_path):
    client, fake = _client(tmp_path)
    r = client.post("/api/ask", json={"question": "che si dice?", "max_steps": 3})
    assert r.status_code == 200
    body = r.json()
    assert "Ciao mondo" in body["answer"]
    assert body["citations"][0]["id"] == MSG_ID
    assert body["tools_used"] == ["semantic_search"]
    assert fake.calls[0]["question"] == "che si dice?"
    assert fake.calls[0]["max_steps"] == 3


def test_ask_requires_question(tmp_path):
    client, _ = _client(tmp_path)
    assert client.post("/api/ask", json={}).status_code == 400


def _parse_sse(text: str) -> list[dict]:
    out = []
    for block in text.split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data: "):
                out.append(json.loads(line[len("data: "):]))
    return out


def test_ask_stream(tmp_path):
    client, _ = _client(tmp_path)
    r = client.get("/api/ask/stream", params={"q": "che si dice?", "chat": "c"})
    assert r.status_code == 200
    events = _parse_sse(r.text)
    tokens = "".join(e["text"] for e in events if e["type"] == "token")
    assert tokens == "Ciao mondo"
    assert any(e["type"] == "event" and e["event"] == "tool_call" for e in events)
    done = [e for e in events if e["type"] == "done"]
    assert done and done[0]["citations"][0]["id"] == MSG_ID
    assert done[0]["answer"].startswith("Ciao mondo")
