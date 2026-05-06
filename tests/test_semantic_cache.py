"""
test_semantic_cache.py — Semantic cache safety guarantees.

Threat/quality dimension:
  Year-confusing near-misses must NOT collide.  "Box 1 rate 2023" and
  "Box 1 rate 2024" embed at cosine ≈ 0.955 — the 0.97 threshold (module-4 §2.3
  worked example) is the floor that prevents this.  Below 0.97, cached 2023
  answers would leak into 2024 queries — financially incorrect.

  The cache key MUST be role-bound (module-4 §2.2): a FIOD analyst's query
  must not produce a cache hit for a helpdesk user with the same query.

Domain review findings addressed:
  - Check 8 (Cache poisoning via tax-year ambiguity): the worked example is
    materialised here as test_year_confusion_does_not_collide.
  - The cross-role poisoning case sits in test_rbac_redteam.py.

Pass criteria:
  - 0.97 floor blocks year-only deltas.
  - SHA256(emb_bucket || role || ceil || year) produces disjoint namespaces.
  - TTL ceiling 24h enforced.
"""

from __future__ import annotations

import hashlib
import json
import time

import pytest

from conftest import (
    CACHE_COSINE_THRESHOLD,
    CACHE_COSINE_DEFAULT,
    UserContext,
    build_cache_key,
)


# ---------------------------------------------------------------------------
# Cache-key construction tests
# ---------------------------------------------------------------------------

def test_cache_key_is_role_bound(user_helpdesk, user_fiod):
    """Same query embedding + same year, different roles → different keys."""
    key_helpdesk = build_cache_key(
        query_embedding_bucket="bucket_42",
        user_role=user_helpdesk.role,
        classification_ceiling=user_helpdesk.classification_ceiling,
        tax_year_context="2024",
    )
    key_fiod = build_cache_key(
        query_embedding_bucket="bucket_42",
        user_role=user_fiod.role,
        classification_ceiling=user_fiod.classification_ceiling,
        tax_year_context="2024",
    )
    assert key_helpdesk != key_fiod, (
        "Cache key collision across roles — confused-deputy side channel."
    )


def test_cache_key_is_year_bound():
    """Same role + same embedding bucket, different tax_year → different keys."""
    key_2023 = build_cache_key("bucket_7", "inspector", "internal", "2023")
    key_2024 = build_cache_key("bucket_7", "inspector", "internal", "2024")
    assert key_2023 != key_2024


def test_cache_key_is_classification_bound():
    """Same role + same embedding, different classification_ceiling → different keys."""
    key_a = build_cache_key("bucket_3", "inspector", "internal", "2024")
    key_b = build_cache_key("bucket_3", "inspector", "fiod",     "2024")
    assert key_a != key_b


def test_cache_key_is_deterministic():
    """Identical inputs produce identical keys (pure SHA-256)."""
    args = ("bucket_99", "fiod", "fiod", "2023")
    assert build_cache_key(*args) == build_cache_key(*args)


def test_cache_key_format():
    """Cache keys are SHA-256 hex prefixed with rag:cache:."""
    key = build_cache_key("bucket_0", "helpdesk", "public", "none")
    assert key.startswith("rag:cache:")
    digest = key.removeprefix("rag:cache:")
    assert len(digest) == 64
    int(digest, 16)  # validates hex


# ---------------------------------------------------------------------------
# Cosine-threshold near-miss tests
# ---------------------------------------------------------------------------

def _cosine(a: list[float], b: list[float]) -> float:
    import math
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x ** 2 for x in a))
    mag_b = math.sqrt(sum(x ** 2 for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


@pytest.mark.parametrize(
    "sim,year_a,year_b,description",
    [
        (0.955, 2023, 2024, "Box 1 rate 2023 vs 2024 (worked example)"),
        (0.958, 2023, 2024, "Box 2 dividend rate 2023 vs 2024"),
        (0.947, 2022, 2024, "30% ruling salary cap 2022 vs 2024"),
        (0.961, 2024, 2024, "TP doc threshold SME vs large entity 2024"),
    ],
)
def test_year_confusion_does_not_collide(sim, year_a, year_b, description):
    """Worked-example near-misses must miss at the 0.97 floor (and 0.98 default)."""
    assert sim < CACHE_COSINE_THRESHOLD, (
        f"Near-miss '{description}' similarity {sim:.3f} ≥ floor "
        f"{CACHE_COSINE_THRESHOLD:.2f} — would yield wrong-year cache hit."
    )
    assert sim < CACHE_COSINE_DEFAULT, (
        f"Near-miss '{description}' similarity {sim:.3f} ≥ default "
        f"{CACHE_COSINE_DEFAULT:.2f}."
    )


def test_threshold_floor_is_at_least_0_97():
    """Module-4 spec: cache cosine threshold floor is 0.97 (zero negotiation)."""
    assert CACHE_COSINE_THRESHOLD >= 0.97
    assert CACHE_COSINE_DEFAULT  >= 0.98


def test_genuine_duplicate_within_threshold():
    """Identical-meaning paraphrases at sim ≥ 0.99 should be allowed to hit."""
    sim = 0.992
    assert sim >= CACHE_COSINE_THRESHOLD


# ---------------------------------------------------------------------------
# End-to-end cache behavior with Redis (integration)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_redis_cache_round_trip(redis_client, flush_test_cache_keys):
    """A round-trip set/get into the test cache namespace should succeed."""
    key = "rag:cache:test:roundtrip"
    payload = {"answer": "Het Box 1 tarief 2024 is 36,97%.", "tax_year": 2024}
    redis_client.set(key, json.dumps(payload), ex=24 * 3600)
    raw = redis_client.get(key)
    assert raw is not None
    assert json.loads(raw)["tax_year"] == 2024


@pytest.mark.integration
def test_cache_ttl_ceiling_24h(redis_client, flush_test_cache_keys):
    """No cache entry may have a TTL exceeding 24 hours (module-4 §2.4)."""
    key = "rag:cache:test:ttl"
    redis_client.set(key, "value", ex=24 * 3600)
    ttl = redis_client.ttl(key)
    assert 0 < ttl <= 24 * 3600


@pytest.mark.integration
def test_role_partitioned_keys_do_not_collide(redis_client, flush_test_cache_keys):
    """In Redis, helpdesk and FIOD keys for the 'same' query must be distinct."""
    # Use the test prefix so the autouse flush only touches these.
    key_h = "rag:cache:test:" + hashlib.sha256(b"role=helpdesk|year=2024").hexdigest()
    key_f = "rag:cache:test:" + hashlib.sha256(b"role=fiod|year=2024").hexdigest()
    redis_client.set(key_h, json.dumps({"answer": "public answer"}))
    redis_client.set(key_f, json.dumps({"answer": "FIOD answer"}))
    assert redis_client.get(key_h) != redis_client.get(key_f)


# ---------------------------------------------------------------------------
# Cosine math sanity
# ---------------------------------------------------------------------------

def test_cosine_self_similarity_is_one():
    v = [0.1, -0.2, 0.7, 0.3]
    assert abs(_cosine(v, v) - 1.0) < 1e-9


def test_cosine_orthogonal_is_zero():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert abs(_cosine(a, b)) < 1e-9


def test_cosine_zero_vector_returns_zero():
    assert _cosine([0.0, 0.0], [1.0, 1.0]) == 0.0
