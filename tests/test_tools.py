from __future__ import annotations

from chat_rag.rag import tools


def test_list_chats(ctx):
    chats = tools.list_chats(ctx)
    assert len(chats) == 1
    assert chats[0]["chat_id"] == "c"
    assert chats[0]["messages"] == 5
    assert chats[0]["is_group"] is False


def test_keyword_search(ctx):
    rows = tools.keyword_search(ctx, "domenica")
    assert any("mare" in r["text"] and r["sender"] == "Alice" for r in rows)
    assert all(r["id"] for r in rows)


def test_semantic_search(ctx):
    rows = tools.semantic_search(ctx, "mare", top=3)
    assert rows
    assert any("mare" in r["text"] for r in rows)
    assert all("score" in r and "id" in r for r in rows)


def test_semantic_search_sender_filter(ctx):
    rows = tools.semantic_search(ctx, "mare", top=5, sender="Bob")
    assert all(r["sender"] == "Bob" for r in rows)


def test_get_stats_per_sender(ctx):
    data = tools.get_stats(ctx, "per_sender")
    names = {r["name"]: r for r in data["rows"]}
    assert names["Alice"]["messages"] == 3
    assert names["Bob"]["messages"] == 2


def test_get_stats_unknown_metric(ctx):
    data = tools.get_stats(ctx, "nope")
    assert "error" in data


def test_get_context(ctx):
    data = tools.get_context(ctx, f"{3:016x}", before=2, after=2)
    ids = [m["id"] for m in data["messages"]]
    assert f"{3:016x}" in ids
    assert f"{1:016x}" in ids
    assert f"{5:016x}" in ids


def test_get_context_missing(ctx):
    data = tools.get_context(ctx, "ffffffffffffffff")
    assert "error" in data


def test_dispatch_unknown_tool(ctx):
    assert "error" in tools.dispatch(ctx, "does_not_exist", {})


def test_collect_message_ids():
    result = {
        "messages": [{"id": "aaaa", "text": "x"}, {"id": "bbbb", "text": "y"}],
        "nested": {"id": "cccc", "text": "z"},
    }
    into: dict = {}
    tools.collect_message_ids(result, into)
    assert set(into) == {"aaaa", "bbbb", "cccc"}
