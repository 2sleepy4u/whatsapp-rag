from __future__ import annotations

from chat_rag.analytics.clustering import ClusterExample
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


def test_semantic_search_context_expands_neighbors(ctx):
    rows = tools.semantic_search(ctx, "mare", top=1, context=2)
    assert len(rows) == 1
    hit = rows[0]
    assert hit["context"]
    assert all(c["id"] != hit["id"] for c in hit["context"])
    assert all(len(c["id"]) == 16 for c in hit["context"])


def test_hybrid_search_merges_and_dedupes(ctx):
    rows = tools.hybrid_search(ctx, "dimenticare", top=5)
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids))
    assert all("channels" in r and "score" in r for r in rows)
    target = f"{3:016x}"
    assert target in ids
    hit = next(r for r in rows if r["id"] == target)
    assert "keyword" in hit["channels"]


def test_hybrid_search_context(ctx):
    rows = tools.hybrid_search(ctx, "mare", top=2, context=1)
    assert rows
    assert all("context" in r for r in rows)


def test_dispatch_hybrid(ctx):
    rows = tools.dispatch(ctx, "hybrid_search", {"query": "mare"})
    assert isinstance(rows, list) and rows


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


def test_dispatch_phrase_timeline(ctx):
    data = tools.dispatch(ctx, "phrase_timeline", {"phrase": "mare"})
    assert data["total"] == 2
    assert all(e["id"] for e in data["evidence"])


def test_dispatch_inside_jokes(ctx):
    data = tools.dispatch(ctx, "inside_joke_candidates", {"min_count": 1, "top": 5})
    assert "candidates" in data
    assert all("phrase" in c and "count" in c for c in data["candidates"])


def test_dispatch_topic_clusters(ctx):
    data = tools.dispatch(ctx, "topic_clusters", {})
    assert "clusters" in data
    assert data["total_windows"] >= 0


def test_window_search_returns_real_messages(ctx):
    rows = tools.window_search(ctx, "mare", top=3)
    assert rows
    first = rows[0]
    assert first["window_id"]
    assert first["messages"]
    valid = {f"{i:016x}" for i in range(1, 6)}
    assert all(m["id"] in valid for m in first["messages"])


def test_dispatch_window_search(ctx):
    rows = tools.dispatch(ctx, "window_search", {"query": "mare", "top": 2})
    assert isinstance(rows, list) and rows


def test_cluster_examples_resolve_real_messages(ctx):
    example = ClusterExample(
        id="w1", date="2024-03-01", sender="", text="window text",
        window_id="w1", message_ids=[f"{1:016x}", f"{2:016x}"],
    )
    out = tools._cluster_examples(ctx, [example])
    assert out[0]["id"] == f"{1:016x}"
    assert out[0]["window_id"] == "w1"
    assert len(out[0]["messages"]) == 2

