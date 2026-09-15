from __future__ import annotations

from pathlib import Path

import pytest

from chat_rag.ingest.parser_android import chat_id_from_filename, parse_file

FIXTURE = Path(__file__).parent / "fixtures" / "android_chat.txt"


@pytest.fixture()
def messages():
    return [msg for msg, _ in parse_file(FIXTURE, tz_name="Europe/Rome")]


def test_message_count(messages):
    assert len(messages) == 11


def test_multiline_continuation(messages):
    msg = next(m for m in messages if m.text.startswith("Alla grande"))
    assert msg.text == "Alla grande\ncontinua qui"
    assert msg.sender == "Marco"


def test_system_encryption_notice(messages):
    msg = next(m for m in messages if "crittografia" in m.text)
    assert msg.msg_type == "system"
    assert msg.sender is None


def test_media_omitted(messages):
    msg = next(m for m in messages if "Media omessi" in m.text)
    assert msg.msg_type == "media"
    assert msg.sender == "Marco"


def test_voice_note_from_attachment(messages):
    msg = next(m for m in messages if "PTT-" in m.text)
    assert msg.msg_type == "voice"
    assert msg.media_ref == "PTT-20230417-WA0001.opus"


def test_deleted(messages):
    msg = next(m for m in messages if "eliminato" in m.text)
    assert msg.msg_type == "deleted"
    assert msg.sender == "Marco"


def test_edited_strips_marker(messages):
    msg = next(m for m in messages if "testo modificato" in m.text)
    assert msg.msg_type == "edited"
    assert msg.text == "testo modificato"


def test_twelve_hour_clock(messages):
    msg = next(m for m in messages if "formato 12h" in m.text)
    assert msg.local_time == "14:15"
    assert msg.local_date == "2023-04-17"


def test_call(messages):
    msg = next(m for m in messages if "Chiamata" in m.text)
    assert msg.msg_type == "call"


def test_sender_with_colon_in_text(messages):
    msg = next(m for m in messages if "due punti" in m.text)
    assert msg.sender == "Marco"
    assert msg.text == "guarda: questo ha i due punti"


def test_chat_id_from_filename():
    assert chat_id_from_filename(Path("WhatsApp Chat with Marco Rossi.txt")) == "Marco Rossi"
    assert chat_id_from_filename(Path("Chat WhatsApp con La mia ragazza.txt")) == "La mia ragazza"
    assert chat_id_from_filename(Path("Gruppo.txt")) == "Gruppo"
