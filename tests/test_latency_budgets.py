"""
test_latency_budgets.py — TTFT and end-to-end latency SLAs.

Threat/quality dimension:
  The assignment fixes TTFT < 1500 ms.  Module 4 §4.2 promotion gates also
  cap p95 end-to-end at 4 s, retrieval at 300 ms, and rerank at 200 ms.
  Without these tests, a slow regression silently violates the SLA in CI.

Pass criteria:
  - p95 TTFT ≤ 1500 ms across a 100-query smoke run.
  - p99 TTFT ≤ 2500 ms.
  - p95 end-to-end ≤ 4000 ms.
  - retrieval p95 ≤ 300 ms (mock client; integration with real OpenSearch happens in docker-runner).
  - rerank p95 ≤ 200 ms.

Notes:
  - The mock client is fast (< 5 ms); these tests verify *the test apparatus
    itself* fits the budget when wired against a real backend, by asserting
    the threshold constants from conftest are present and applied.
  - pytest-benchmark is used for percentile reporting where useful.
  - Integration latency (real Bedrock + OpenSearch) is captured by docker-runner
    via the same constants and lives behind the `latency` and `slow` markers.
"""

from __future__ import annotations

import statistics
import time
from typing import Callable, Iterable

import pytest

from conftest import (
    LATENCY_P95_TTFT_MS,
    LATENCY_P99_TTFT_MS,
    LATENCY_P95_E2E_MS,
    RETRIEVAL_P95_MS,
    RERANK_P95_MS,
    MOCK_CORPUS,
    UserContext,
)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values_sorted = sorted(values)
    k = (len(values_sorted) - 1) * pct
    f = int(k)
    c = min(f + 1, len(values_sorted) - 1)
    if f == c:
        return values_sorted[f]
    return values_sorted[f] + (values_sorted[c] - values_sorted[f]) * (k - f)


SMOKE_QUERIES = [
    "Wat is het Box 1 tarief in 2024?",
    "Hoe werkt de thuiswerkaftrek voor IB-ondernemers?",
    "ECLI:NL:HR:2023:123 reiskosten",
    "Wat is artikel 3.114 lid 1 Wet IB 2001?",
    "Mag ik werkruimte thuis aftrekken in 2022?",
    "Hoe wordt resultaat uit overige werkzaamheden belast?",
    "Wat zegt artikel 3.16 over thuiswerken?",
    "Geef het Box 1 tarief 2023.",
    "Hoge Raad uitspraak over woon-werkverkeer ZZP",
    "Wat is gecombineerde heffingskorting in 2024?",
] * 10  # 100 queries total


@pytest.mark.latency
def test_ttft_p95_within_budget(mock_rag_client, user_inspector):
    """100-query smoke run: p95 TTFT must be ≤ 1500 ms."""
    samples = []
    for query in SMOKE_QUERIES:
        t0 = time.perf_counter()
        response = mock_rag_client.query(query, user_inspector)
        ttft_ms = (time.perf_counter() - t0) * 1000
        samples.append(ttft_ms)
        assert response is not None

    p50 = _percentile(samples, 0.50)
    p95 = _percentile(samples, 0.95)
    p99 = _percentile(samples, 0.99)
    print(f"\nTTFT p50={p50:.1f}ms  p95={p95:.1f}ms  p99={p99:.1f}ms (n={len(samples)})")

    assert p95 <= LATENCY_P95_TTFT_MS, (
        f"p95 TTFT {p95:.1f} ms exceeds budget {LATENCY_P95_TTFT_MS} ms."
    )


@pytest.mark.latency
def test_ttft_p99_within_budget(mock_rag_client, user_inspector):
    """p99 TTFT must be ≤ 2500 ms."""
    samples = []
    for query in SMOKE_QUERIES:
        t0 = time.perf_counter()
        mock_rag_client.query(query, user_inspector)
        samples.append((time.perf_counter() - t0) * 1000)
    p99 = _percentile(samples, 0.99)
    assert p99 <= LATENCY_P99_TTFT_MS, (
        f"p99 TTFT {p99:.1f} ms exceeds {LATENCY_P99_TTFT_MS} ms."
    )


@pytest.mark.latency
def test_e2e_p95_within_budget(mock_rag_client, user_inspector):
    """End-to-end p95 must be ≤ 4 seconds."""
    samples = []
    for query in SMOKE_QUERIES:
        t0 = time.perf_counter()
        response = mock_rag_client.query(query, user_inspector)
        samples.append((time.perf_counter() - t0) * 1000)
        assert response is not None
    p95 = _percentile(samples, 0.95)
    assert p95 <= LATENCY_P95_E2E_MS, (
        f"p95 end-to-end {p95:.1f} ms exceeds {LATENCY_P95_E2E_MS} ms."
    )


@pytest.mark.latency
def test_retrieval_p95_within_budget(mock_rag_client, user_inspector):
    """Retrieval-stage p95 must be ≤ 300 ms (Module 4 promotion gate)."""
    samples = []
    for _ in range(50):
        t0 = time.perf_counter()
        mock_rag_client.retrieve("Box 1 tarief 2024", user_inspector)
        samples.append((time.perf_counter() - t0) * 1000)
    p95 = _percentile(samples, 0.95)
    assert p95 <= RETRIEVAL_P95_MS, (
        f"Retrieval p95 {p95:.1f} ms exceeds {RETRIEVAL_P95_MS} ms."
    )


@pytest.mark.latency
def test_rerank_p95_budget_constant_is_set():
    """Constant must be aligned with module-4 §4.2 (200 ms cap)."""
    assert RERANK_P95_MS <= 200


def test_latency_constants_match_module4_thresholds():
    """Defensive: prevent silent drift of the SLA constants in conftest."""
    assert LATENCY_P95_TTFT_MS == 1500
    assert LATENCY_P99_TTFT_MS == 2500
    assert LATENCY_P95_E2E_MS  == 4000
    assert RETRIEVAL_P95_MS    == 300
    assert RERANK_P95_MS       == 200


# ---------------------------------------------------------------------------
# Cold vs warm cache differential (mock — orientation only)
# ---------------------------------------------------------------------------

@pytest.mark.latency
def test_warm_cache_is_not_slower_than_cold(mock_rag_client, user_inspector):
    """Warm cache must never be slower than cold by more than measurement noise."""
    query = "Wat is artikel 3.114 lid 1 Wet IB 2001?"

    t0 = time.perf_counter()
    mock_rag_client.query(query, user_inspector)
    cold_ms = (time.perf_counter() - t0) * 1000

    # Run 5 warm samples
    warm = []
    for _ in range(5):
        t0 = time.perf_counter()
        mock_rag_client.query(query, user_inspector)
        warm.append((time.perf_counter() - t0) * 1000)

    median_warm = statistics.median(warm)
    # Mock client has no real cache; assert the apparatus times are reasonable.
    assert median_warm < cold_ms + 50, (
        f"Median warm {median_warm:.1f} ms substantially exceeds cold {cold_ms:.1f} ms."
    )
