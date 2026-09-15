from __future__ import annotations

import pytest

from chat_rag.config import Settings
from chat_rag.db.db import open_db
from chat_rag.transcribe.base import TranscriptResult, build_engine
from chat_rag.transcribe.pipeline import media_search_roots, transcribe_pending, voice_messages
from chat_rag.transcribe.whisper_cpp_engine import WhisperCppEngine

CHAT = "c"
A = "c::Luca"
VOICE_FILE = "PTT-20240101-WA0001.opus"


class FakeEngine:
    name = "fake"
    model = "fake-model"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def transcribe(self, path):
        self.calls.append(path.name)
        return TranscriptResult(text="ciao questa è una nota vocale", language="it", duration=3.2)


def _seed(tmp_path):
    exports = tmp_path / "exports"
    (exports / "Media").mkdir(parents=True)
    (exports / "Media" / VOICE_FILE).write_bytes(b"fake-opus")
    (exports / "chat.txt").write_text("x", encoding="utf-8")

    conn = open_db(tmp_path / "chat.db")
    conn.execute("INSERT INTO chats (chat_id, name) VALUES (?, ?)", (CHAT, CHAT))
    conn.execute("INSERT INTO senders (sender_id, chat_id, name) VALUES (?, ?, 'Luca')", (A, CHAT))
    rows = [
        ("v1", VOICE_FILE),
        ("v2", "PTT-20240101-WA9999.opus"),
    ]
    for i, (mid, ref) in enumerate(rows):
        conn.execute(
            """
            INSERT INTO messages
                (id, chat_id, sender_id, ts, ts_local, local_date, local_time, text,
                 raw_line, msg_type, media_ref, source_file, line_no, ingested_at)
            VALUES (?, ?, ?, ?, '2024-01-01 10:00:00', '2024-01-01', '10:00', ?, ?, 'voice', ?, ?, ?, 0)
            """,
            (mid, CHAT, A, 1700000000 + i, f"{ref} (file allegato)", f"{ref} (file allegato)",
             ref, str(exports / "chat.txt"), i),
        )
    conn.execute(
        """
        INSERT INTO messages
            (id, chat_id, sender_id, ts, ts_local, local_date, local_time, text,
             raw_line, msg_type, source_file, line_no, ingested_at)
        VALUES ('t1', ?, ?, 1700000100, '2024-01-01 10:01:00', '2024-01-01', '10:01',
                'testo normale', 'testo normale', 'text', ?, 9, 0)
        """,
        (CHAT, A, str(exports / "chat.txt")),
    )
    conn.commit()
    return conn, exports


def test_voice_messages(tmp_path):
    conn, _ = _seed(tmp_path)
    rows = voice_messages(conn)
    assert [r["id"] for r in rows] == ["v1", "v2"]


def test_dry_run_does_not_engine(tmp_path):
    conn, exports = _seed(tmp_path)
    roots = media_search_roots(conn)
    stats = transcribe_pending(conn, None, roots, dry_run=True)
    assert stats.total == 2
    assert stats.transcribed == 1  # v1 file exists
    assert stats.missing == 1  # v2 file absent
    assert conn.execute("SELECT COUNT(*) FROM transcripts").fetchone()[0] == 0
    assert "PTT" in conn.execute("SELECT text FROM messages WHERE id='v1'").fetchone()[0]


def test_transcribe_and_apply(tmp_path):
    conn, exports = _seed(tmp_path)
    roots = media_search_roots(conn)
    engine = FakeEngine()
    stats = transcribe_pending(conn, engine, roots, apply_to_messages=True)
    assert stats.transcribed == 1
    assert stats.missing == 1
    assert engine.calls == [VOICE_FILE]

    transcript = conn.execute("SELECT * FROM transcripts WHERE message_id='v1'").fetchone()
    assert transcript["text"] == "ciao questa è una nota vocale"
    assert transcript["engine"] == "fake"
    assert transcript["language"] == "it"

    updated = conn.execute("SELECT text, msg_type FROM messages WHERE id='v1'").fetchone()
    assert updated["text"] == "ciao questa è una nota vocale"
    assert updated["msg_type"] == "voice"

    # FTS reflects the transcript after the UPDATE trigger fired.
    hits = conn.execute(
        "SELECT m.id FROM messages_fts f JOIN messages m ON m.rowid=f.rowid "
        "WHERE messages_fts MATCH 'vocale'"
    ).fetchall()
    assert any(r["id"] == "v1" for r in hits)


def test_rerun_skips_and_overwrite(tmp_path):
    conn, _ = _seed(tmp_path)
    roots = media_search_roots(conn)
    transcribe_pending(conn, FakeEngine(), roots)

    again = transcribe_pending(conn, FakeEngine(), roots)
    assert again.skipped == 1
    assert again.transcribed == 0

    forced = transcribe_pending(conn, FakeEngine(), roots, overwrite=True)
    assert forced.transcribed == 1


def test_keep_placeholder(tmp_path):
    conn, _ = _seed(tmp_path)
    roots = media_search_roots(conn)
    transcribe_pending(conn, FakeEngine(), roots, apply_to_messages=False)
    assert "PTT" in conn.execute("SELECT text FROM messages WHERE id='v1'").fetchone()[0]
    assert conn.execute("SELECT COUNT(*) FROM transcripts").fetchone()[0] == 1


def test_whisper_cpp_missing_binary():
    with pytest.raises(FileNotFoundError):
        WhisperCppEngine(bin_path="/nope/whisper-cli", model_path="/nope/model.bin")


def _settings(tmp_path, **overrides) -> Settings:
    base = dict(
        data_dir=tmp_path, tz="Europe/Rome", ollama_host="http://localhost:11434",
        llm_model="m", embed_model="e", llm_num_predict=64, llm_think=False,
        transcribe_engine="auto", whisper_model="small", whisper_language="it",
        whisper_device="cpu", whisper_compute_type="int8",
        whisper_cpp_bin=None, whisper_cpp_model=None,
    )
    base.update(overrides)
    return Settings(**base)


def test_build_engine_prefers_cpp_when_configured(tmp_path):
    settings = _settings(tmp_path, whisper_cpp_bin="/nope/bin", whisper_cpp_model="/nope/model.bin")
    with pytest.raises(FileNotFoundError):
        build_engine(settings)
