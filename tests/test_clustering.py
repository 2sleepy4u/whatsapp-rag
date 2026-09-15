from __future__ import annotations

from chat_rag.analytics.clustering import analyze_topics, cluster_records, fetch_window_records
from chat_rag.embed.index import chroma_client, ensure_collection

DIM = 8


def _rec(i, group, date, text):
    vec = [0.0] * DIM
    vec[group] = 1.0
    vec[(group + 1) % DIM] = 0.08 * (i % 3)
    return {
        "id": f"w{i}",
        "text": text,
        "embedding": vec,
        "meta": {"start_date": date, "sender_name": "Alice" if i % 2 else "Bob"},
    }


def _records():
    return [
        _rec(0, 0, "2024-01-05", "andiamo al mare domenica"),
        _rec(1, 0, "2024-01-20", "mare bellissimo oggi"),
        _rec(2, 0, "2024-02-02", "acqua fredda al mare"),
        _rec(3, 1, "2024-02-10", "esame di matematica difficile"),
        _rec(4, 1, "2024-03-01", "studio matematica tutta notte"),
        _rec(5, 1, "2024-03-15", "matematica passata finalmente"),
        _rec(6, 1, "2024-04-01", "ripasso matematica di nuovo"),
    ]


def test_cluster_records_finds_two_topics():
    result = cluster_records(_records(), min_cluster_size=2, top_terms=5, top_examples=2)
    assert len(result.clusters) >= 2
    assert result.total == 7
    assert sum(c.size for c in result.clusters) == 7
    for c in result.clusters:
        assert c.examples
        assert c.terms
    all_terms = {t for c in result.clusters for t in c.terms}
    assert "mare" in all_terms
    assert "matematica" in all_terms


def test_cluster_too_few_records():
    result = cluster_records(_records()[:1], min_cluster_size=2)
    assert result.clusters == []
    assert result.labels == [-1]


def test_chroma_wrapper_and_evolution(tmp_path):
    client = chroma_client(tmp_path / "chroma")
    col = ensure_collection(client, "messages")
    recs = _records()
    col.add(
        ids=[r["id"] for r in recs],
        embeddings=[r["embedding"] for r in recs],
        documents=[r["text"] for r in recs],
        metadatas=[{"kind": "window", "chat_id": "c", **r["meta"]} for r in recs],
    )
    records = fetch_window_records(col, chat_id="c")
    assert len(records) == 7

    result, evolution = analyze_topics(
        col, chat_id="c", min_cluster_size=2, evolution_bucket="month", evolution_top=2
    )
    assert len(result.clusters) >= 2
    all_terms = {t for c in result.clusters for t in c.terms}
    assert "mare" in all_terms and "matematica" in all_terms
    assert evolution
    assert all(e["series"] for e in evolution)


def test_fetch_filters_by_chat(tmp_path):
    client = chroma_client(tmp_path / "chroma")
    col = ensure_collection(client, "messages")
    recs = _records()
    col.add(
        ids=[r["id"] for r in recs],
        embeddings=[r["embedding"] for r in recs],
        documents=[r["text"] for r in recs],
        metadatas=[{"kind": "window", "chat_id": "a", **r["meta"]} for r in recs],
    )
    assert fetch_window_records(col, chat_id="b") == []
    assert len(fetch_window_records(col, chat_id="a")) == 7
