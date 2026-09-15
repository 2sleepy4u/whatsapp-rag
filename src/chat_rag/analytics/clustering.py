"""Topic discovery by clustering conversation-window embeddings.

Windows (see ``embed.index``) are short sequences of consecutive messages. We
normalise their embeddings, optionally reduce dimensionality with PCA, cluster
with HDBSCAN, then describe every cluster by its most distinctive terms
(c-TF-IDF) and its most central messages (usable as citations).

``cluster_records`` is pure (takes vectors) and therefore testable; the
``topic_clusters`` wrapper fetches windows from a Chroma collection.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from ..db.stats import ITALIAN_STOPWORDS

_WORD_RE = re.compile(r"[0-9a-zàáèéìíòóùúâêîôûäëïöüç']+", re.IGNORECASE)


@dataclass
class ClusterExample:
    id: str
    date: str
    sender: str
    text: str


@dataclass
class Cluster:
    cluster_id: int
    size: int
    terms: list[str] = field(default_factory=list)
    start_date: str = ""
    end_date: str = ""
    examples: list[ClusterExample] = field(default_factory=list)

    @property
    def label(self) -> str:
        return ", ".join(self.terms[:3]) or f"cluster {self.cluster_id}"


@dataclass
class ClusteringResult:
    clusters: list[Cluster]
    total: int
    noise: int
    labels: list[int] = field(default_factory=list)

    @property
    def noise_ratio(self) -> float:
        return self.noise / self.total if self.total else 0.0

    @property
    def terms_by_cluster(self) -> dict[int, list[str]]:
        return {c.cluster_id: c.terms for c in self.clusters}


def _tokens(text: str) -> list[str]:
    return [w for w in _WORD_RE.findall(text.lower()) if len(w) >= 3 and w not in ITALIAN_STOPWORDS]


def _ctfidf_terms(docs_by_cluster: dict[int, list[str]], top: int) -> dict[int, list[str]]:
    cluster_tf: dict[int, Counter] = {}
    global_df: Counter = Counter()
    for cid, docs in docs_by_cluster.items():
        tf: Counter = Counter()
        seen: set[str] = set()
        for doc in docs:
            for tok in _tokens(doc):
                tf[tok] += 1
                seen.add(tok)
        cluster_tf[cid] = tf
        for tok in seen:
            global_df[tok] += 1

    n_clusters = max(1, len(docs_by_cluster))
    total_words = sum(sum(tf.values()) for tf in cluster_tf.values()) or 1
    avg_words = total_words / n_clusters

    out: dict[int, list[str]] = {}
    for cid, tf in cluster_tf.items():
        scored = []
        for tok, count in tf.items():
            weight = count * math.log(1 + avg_words / max(1, global_df[tok]))
            scored.append((weight, tok))
        scored.sort(reverse=True)
        out[cid] = [tok for _, tok in scored[:top]]
    return out


def cluster_records(
    records: list[dict],
    min_cluster_size: int = 5,
    reduce_dim: int = 48,
    top_terms: int = 8,
    top_examples: int = 3,
    random_state: int = 42,
) -> ClusteringResult:
    """records: dicts with keys ``id``, ``text``, ``embedding`` and ``meta``."""
    if len(records) < max(2, min_cluster_size):
        return ClusteringResult(clusters=[], total=len(records), noise=len(records), labels=[-1] * len(records))

    vectors = np.asarray([r["embedding"] for r in records], dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = vectors / norms

    reduced = vectors
    if reduce_dim and vectors.shape[1] > reduce_dim and len(records) > reduce_dim + 1:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=reduce_dim, svd_solver="randomized", random_state=random_state)
        reduced = pca.fit_transform(vectors)
        rnorms = np.linalg.norm(reduced, axis=1, keepdims=True)
        rnorms[rnorms == 0] = 1.0
        reduced = reduced / rnorms

    from sklearn.cluster import HDBSCAN

    labels = HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean", copy=True).fit_predict(reduced)

    groups: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        if label == -1:
            continue
        groups.setdefault(int(label), []).append(idx)

    docs_by_cluster = {cid: [records[i]["text"] for i in idxs] for cid, idxs in groups.items()}
    terms_by_cluster = _ctfidf_terms(docs_by_cluster, top_terms)

    clusters: list[Cluster] = []
    for cid, idxs in groups.items():
        centroid = reduced[idxs].mean(axis=0)
        centroid /= np.linalg.norm(centroid) or 1.0
        dists = np.linalg.norm(reduced[idxs] - centroid, axis=1)
        order = np.argsort(dists)[:top_examples]
        examples = []
        dates = []
        for pos in order:
            rec = records[idxs[int(pos)]]
            meta = rec["meta"]
            date = meta.get("start_date") or meta.get("local_date", "")
            dates.append(date)
            examples.append(
                ClusterExample(
                    id=rec["id"],
                    date=date,
                    sender=meta.get("sender_name", ""),
                    text=rec["text"][:300],
                )
            )
        all_dates = [records[i]["meta"].get("start_date", "") for i in idxs if records[i]["meta"].get("start_date")]
        clusters.append(
            Cluster(
                cluster_id=cid,
                size=len(idxs),
                terms=terms_by_cluster.get(cid, []),
                start_date=min(all_dates) if all_dates else "",
                end_date=max(all_dates) if all_dates else "",
                examples=examples,
            )
        )

    clusters.sort(key=lambda c: c.size, reverse=True)
    noise = int(np.sum(labels == -1))
    return ClusteringResult(clusters=clusters, total=len(records), noise=noise, labels=[int(x) for x in labels])


# --- Chroma wrapper --------------------------------------------------------


def fetch_window_records(collection, chat_id: str | None = None,
                         date_from: str | None = None, date_to: str | None = None) -> list[dict]:
    data = collection.get(include=["embeddings", "documents", "metadatas"])
    ids = data.get("ids") or []
    embeddings = data.get("embeddings")
    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or []
    if embeddings is None:
        return []

    records: list[dict] = []
    for i, mid in enumerate(ids):
        meta = metadatas[i] or {}
        if meta.get("kind") != "window":
            continue
        if chat_id and meta.get("chat_id") != chat_id:
            continue
        start = meta.get("start_date", "")
        if date_from and start and start < date_from:
            continue
        if date_to and start and start > date_to:
            continue
        records.append(
            {
                "id": mid,
                "text": documents[i] or "",
                "embedding": list(embeddings[i]),
                "meta": meta,
            }
        )
    return records


def topic_clusters(collection, chat_id: str | None = None,
                   date_from: str | None = None, date_to: str | None = None,
                   min_cluster_size: int = 5, top_terms: int = 8,
                   top_examples: int = 3, reduce_dim: int = 48) -> ClusteringResult:
    records = fetch_window_records(collection, chat_id, date_from, date_to)
    return cluster_records(
        records,
        min_cluster_size=min_cluster_size,
        reduce_dim=reduce_dim,
        top_terms=top_terms,
        top_examples=top_examples,
    )


def analyze_topics(collection, chat_id: str | None = None,
                   date_from: str | None = None, date_to: str | None = None,
                   min_cluster_size: int = 5, top_terms: int = 8,
                   top_examples: int = 3, reduce_dim: int = 48,
                   evolution_bucket: str | None = None,
                   evolution_top: int = 5) -> tuple[ClusteringResult, list[dict]]:
    records = fetch_window_records(collection, chat_id, date_from, date_to)
    result = cluster_records(
        records,
        min_cluster_size=min_cluster_size,
        reduce_dim=reduce_dim,
        top_terms=top_terms,
        top_examples=top_examples,
    )
    evolution: list[dict] = []
    if evolution_bucket and result.labels:
        evolution = topic_evolution(
            records, result.labels, result.terms_by_cluster, bucket=evolution_bucket, top=evolution_top
        )
    return result, evolution


def topic_evolution(records: list[dict], labels: list[int], terms_by_cluster: dict[int, list[str]],
                    bucket: str = "month", top: int = 5) -> list[dict]:
    """Per-period activity of the largest clusters (needs labels from cluster_records)."""
    fmt = {"day": "%Y-%m-%d", "week": "%Y-%W", "month": "%Y-%m", "year": "%Y"}[bucket]
    sizes: Counter = Counter(l for l in labels if l != -1)
    keep = [cid for cid, _ in sizes.most_common(top)]

    from datetime import datetime

    series: dict[int, Counter] = {cid: Counter() for cid in keep}
    for rec, label in zip(records, labels):
        if label not in series:
            continue
        date = rec["meta"].get("start_date") or rec["meta"].get("local_date", "")
        if not date:
            continue
        try:
            period = datetime.strptime(date, "%Y-%m-%d").strftime(fmt)
        except ValueError:
            continue
        series[label][period] += 1

    out = []
    for cid in keep:
        periods = sorted(series[cid])
        out.append(
            {
                "cluster_id": cid,
                "label": ", ".join(terms_by_cluster.get(cid, [])[:3]),
                "size": int(sizes[cid]),
                "series": [{"period": p, "count": series[cid][p]} for p in periods],
            }
        )
    return out
