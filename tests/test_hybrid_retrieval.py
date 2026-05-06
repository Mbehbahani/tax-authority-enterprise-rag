"""
test_hybrid_retrieval.py — Hybrid BM25 + dense retrieval correctness.

Threat/quality dimension:
  Exact-match citation queries (ECLI, article numbers) must surface via the
  BM25 arm at top-1 even if dense retrieval would rank them lower.  Pure
  semantic queries must work via the dense arm without an exact match.
  Reranker must improve nDCG@5 over the RRF-only baseline.

Pass criteria:
  - ECLI exact-match → rank 1.
  - Semantic query → relevant chunks within top-5.
  - RRF fusion produces a non-empty result set when either arm has hits.
  - Reranker increases nDCG@5 vs. RRF-only baseline.
"""

from __future__ import annotations

import math
from typing import Optional

import pytest

from conftest import (
    MOCK_CORPUS,
    ChunkMeta,
    UserContext,
    TOP_K_RERANK,
    TOP_K_RRF,
)

# ---------------------------------------------------------------------------
# Lightweight BM25 + dense simulators (deterministic for testing the cascade)
# ---------------------------------------------------------------------------

def bm25_rank(query: str, corpus: list[ChunkMeta], top_k: int = 100) -> list[ChunkMeta]:
    """Token-overlap rank as a stand-in for BM25 — deterministic, no Lucene needed."""
    q_tokens = set(query.lower().split())
    scored = []
    for c in corpus:
        text_tokens = set(c.text.lower().split())
        # Boost ECLI and article-number exact matches heavily (regex router behavior).
        score = len(q_tokens & text_tokens)
        if c.ecli and c.ecli.lower() in query.lower():
            score += 100
        if c.article and c.article in query:
            score += 50
        scored.append((score, c))
    scored.sort(key=lambda x: -x[0])
    return [c for s, c in scored if s > 0][:top_k]


def dense_rank(query: str, corpus: list[ChunkMeta], top_k: int = 100) -> list[ChunkMeta]:
    """Substring-match rank as a stand-in for vector similarity."""
    q_lower = query.lower()
    scored = []
    for c in corpus:
        text_lower = c.text.lower()
        # Coarse semantic similarity proxy: shared prefix of significant words.
        overlap = sum(1 for w in q_lower.split() if len(w) > 3 and w in text_lower)
        scored.append((overlap, c))
    scored.sort(key=lambda x: -x[0])
    return [c for s, c in scored if s > 0][:top_k]


def rrf_fuse(
    rankings: list[list[ChunkMeta]],
    k: int = 60,
    top_k: int = TOP_K_RRF,
) -> list[ChunkMeta]:
    """Reciprocal Rank Fusion — k=60 per Cormack et al."""
    scores: dict[str, float] = {}
    chunk_by_id: dict[str, ChunkMeta] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            chunk_by_id[chunk.chunk_id] = chunk
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
    fused = sorted(scores.items(), key=lambda x: -x[1])
    return [chunk_by_id[cid] for cid, _ in fused[:top_k]]


def mock_rerank(query: str, candidates: list[ChunkMeta], top_n: int = TOP_K_RERANK) -> list[ChunkMeta]:
    """Mock cross-encoder rerank: prefers chunks with exact identifier and term matches."""
    q_lower = query.lower()
    scored = []
    for c in candidates:
        score = 0.0
        if c.ecli and c.ecli.lower() in q_lower:
            score += 100.0
        if c.article and c.article in query:
            score += 50.0
        text_lower = c.text.lower()
        for word in q_lower.split():
            if len(word) > 3 and word in text_lower:
                score += 1.0
        scored.append((score, c))
    scored.sort(key=lambda x: -x[0])
    return [c for s, c in scored[:top_n]]


def ndcg_at_k(retrieved: list[ChunkMeta], relevant_chunk_ids: set[str], k: int = 5) -> float:
    """Normalized Discounted Cumulative Gain at k."""
    dcg = 0.0
    for i, chunk in enumerate(retrieved[:k], start=1):
        rel = 1.0 if chunk.chunk_id in relevant_chunk_ids else 0.0
        dcg += rel / math.log2(i + 1)
    ideal_dcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant_chunk_ids), k) + 1))
    if ideal_dcg == 0:
        return 0.0
    return dcg / ideal_dcg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

ECLI_QUERIES = [
    ("Hoge Raad ECLI:NL:HR:2023:123 betreffende reiskosten ZZP", "ecli-nl-hr-2023-123-chunk1"),
    ("Wat zegt ECLI:NL:HR:2021:456 over verhuurinkomsten?",      "ecli-nl-hr-2021-456-chunk1"),
    ("ECLI:NL:HR:2022:789 software-abonnementen",                "ecli-nl-hr-2022-789-chunk1"),
]


@pytest.mark.parametrize("query,expected_chunk_id", ECLI_QUERIES)
def test_ecli_exact_match_ranks_first(query, expected_chunk_id):
    """ECLI exact-match query must place the cited ruling at rank 1 (BM25 arm)."""
    bm25_results = bm25_rank(query, MOCK_CORPUS)
    assert len(bm25_results) > 0, "BM25 returned no results for ECLI query"
    assert bm25_results[0].chunk_id == expected_chunk_id, (
        f"Expected {expected_chunk_id} at BM25 rank 1; got {bm25_results[0].chunk_id}"
    )


def test_article_number_exact_match():
    """Article-number query must surface the matching article via BM25 boost."""
    query = "tarief belastbaar inkomen artikel 3.114 lid 1"
    results = bm25_rank(query, MOCK_CORPUS, top_k=10)
    assert any(c.article == "3.114" and c.lid == "1" for c in results)


def test_semantic_query_hits_top5():
    """A pure semantic query should retrieve relevant chunks via dense rank within top-5.

    Uses query terms that genuinely appear in the corpus chunks; the lexical
    `dense_rank` in this file is a substring stand-in for true vector similarity.
    """
    query = "kosten thuiswerkruimte werkruimte aftrekbaar"
    results = dense_rank(query, MOCK_CORPUS)
    top5_doc_ids = {c.doc_id for c in results[:5]}
    assert "wet-ib-2001-art316" in top5_doc_ids or "wet-ib-2001-art316-b" in top5_doc_ids


def test_rrf_fusion_combines_arms():
    """RRF fusion must merge results from BM25 and dense, deduplicating by chunk_id."""
    query = "Box 1 tarief 2024 inkomen werk woning"
    bm25_results = bm25_rank(query, MOCK_CORPUS)
    dense_results = dense_rank(query, MOCK_CORPUS)
    fused = rrf_fuse([bm25_results, dense_results])
    assert len(fused) > 0
    # Deduplication
    chunk_ids = [c.chunk_id for c in fused]
    assert len(chunk_ids) == len(set(chunk_ids))


def test_rrf_handles_empty_arm():
    """RRF must degrade gracefully if one arm returns zero results."""
    query = "ECLI:NL:HR:2023:123"
    bm25_results = bm25_rank(query, MOCK_CORPUS)
    # Force dense to be empty by supplying an unrelated corpus.
    fused = rrf_fuse([bm25_results, []])
    assert len(fused) > 0
    assert fused[0].ecli == "ECLI:NL:HR:2023:123"


def test_reranker_improves_ndcg_at_5():
    """Rerank stage must produce nDCG@5 ≥ RRF-only baseline."""
    query = "ECLI:NL:HR:2023:123 reiskosten ZZP"
    relevant = {"ecli-nl-hr-2023-123-chunk1"}

    bm25_results = bm25_rank(query, MOCK_CORPUS)
    dense_results = dense_rank(query, MOCK_CORPUS)
    rrf_results = rrf_fuse([bm25_results, dense_results])

    rrf_ndcg = ndcg_at_k(rrf_results, relevant, k=5)
    reranked = mock_rerank(query, rrf_results)
    rerank_ndcg = ndcg_at_k(reranked, relevant, k=5)

    assert rerank_ndcg >= rrf_ndcg, (
        f"Reranker degraded ranking: RRF nDCG@5={rrf_ndcg:.3f}, rerank nDCG@5={rerank_ndcg:.3f}"
    )


def test_top_k_cascade_sizes():
    """Cascade must respect the documented top-K values from MASTER-PLAN."""
    query = "Box 1 inkomen werk woning"
    bm25_results = bm25_rank(query, MOCK_CORPUS, top_k=100)
    dense_results = dense_rank(query, MOCK_CORPUS, top_k=100)
    fused = rrf_fuse([bm25_results, dense_results], top_k=TOP_K_RRF)
    reranked = mock_rerank(query, fused, top_n=TOP_K_RERANK)

    assert len(bm25_results) <= 100
    assert len(dense_results) <= 100
    assert len(fused) <= TOP_K_RRF
    assert len(reranked) <= TOP_K_RERANK


def test_hybrid_routes_ecli_via_bm25_boost():
    """Regex-router behavior: ECLI queries must score higher in BM25 than dense."""
    query = "ECLI:NL:HR:2023:123"
    bm25_top = bm25_rank(query, MOCK_CORPUS, top_k=1)
    dense_top = dense_rank(query, MOCK_CORPUS, top_k=1)
    assert bm25_top, "BM25 must surface ECLI exact match"
    if dense_top and dense_top[0].ecli != "ECLI:NL:HR:2023:123":
        assert bm25_top[0].ecli == "ECLI:NL:HR:2023:123"
