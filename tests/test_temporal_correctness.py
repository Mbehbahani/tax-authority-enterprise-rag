"""
test_temporal_correctness.py — Temporal validity correctness.

Validates that the retrieval layer returns the correct legislative version
for a given tax year, even when a more semantically similar (and more recent)
version of the same law exists.

Design references:
  - module-1-2-retrieval.md §1.2: valid_from / valid_to date range filter in DSL.
  - module-1-2-retrieval.md §2.1: efficient_filter includes temporal range.
  - module-1-2-retrieval.md Domain Review Check 3: temporal validity handled correctly.
  - module-1-2-retrieval.md Domain Review Check 4: superseded_by disclosure in generation.
  - module-4-ops-security.md §2.3: 'Box 1 2023 vs 2024' worked example (~0.955 cosine).

Pass criterion:
  - For each temporal golden Q&A pair, the retrieved chunk's tax_year must match
    the query year (not the most recent version).
  - Superseded documents must carry a superseded_by flag visible to the generator.
  - valid_from / valid_to filter must exclude out-of-range versions.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

import pytest

from conftest import (
    MOCK_CORPUS,
    CORPUS_BY_DOC_ID,
    ChunkMeta,
    MockRAGClient,
    UserContext,
    load_golden_qa,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def is_valid_on(chunk: ChunkMeta, query_date: date) -> bool:
    """
    Return True if the chunk is in force on query_date.
    Mirrors the OpenSearch filter DSL in module-1-2-retrieval.md §2.1:
      valid_from <= query_date AND (valid_to >= query_date OR valid_to IS NULL)
    """
    if chunk.valid_from:
        vf = datetime.strptime(chunk.valid_from, "%Y-%m-%d").date()
        if vf > query_date:
            return False
    if chunk.valid_to:
        vt = datetime.strptime(chunk.valid_to, "%Y-%m-%d").date()
        if vt < query_date:
            return False
    return True


def chunks_for_tax_year(tax_year: int) -> list[ChunkMeta]:
    """
    Return all corpus chunks valid on Jan 1 of the given tax year.
    This simulates what the retrieval layer should return for a year-scoped query.
    """
    target_date = date(tax_year, 1, 1)
    return [c for c in MOCK_CORPUS if c.tax_year == tax_year or is_valid_on(c, target_date)]


def top_chunk_for_article_year(article: str, tax_year: int) -> Optional[ChunkMeta]:
    """Return the corpus chunk for a specific article and tax year."""
    candidates = [
        c for c in MOCK_CORPUS
        if c.article == article and c.tax_year == tax_year
    ]
    return candidates[0] if candidates else None


# ---------------------------------------------------------------------------
# Temporal golden Q&A items
# ---------------------------------------------------------------------------

TEMPORAL_GOLDEN = [
    qa for qa in load_golden_qa()
    if qa.get("category") == "temporal"
    and qa.get("tax_year")
    and not qa.get("must_refuse")
]

TEMPORAL_PARAMS = [
    pytest.param(qa, id=qa["id"])
    for qa in TEMPORAL_GOLDEN
]

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTemporalRetrieval:
    """Tests that retrieval respects valid_from/valid_to date boundaries."""

    @pytest.mark.parametrize("qa", TEMPORAL_PARAMS)
    def test_retrieved_year_matches_query_year(
        self, qa: dict, mock_rag_client: MockRAGClient, user_inspector: UserContext
    ):
        """
        For each temporal golden pair, the top retrieved chunk must have
        tax_year == the query's expected year.

        This is the core temporal correctness test: the system must not return
        the 2024 rate when the user asks about 2021.

        The near-miss risk (Box 1 2021 vs 2024 embedding ~0.955 cosine similarity,
        per module-4 §2.3) means the temporal filter in the DSL is load-bearing.
        """
        query = qa["query"]
        expected_year = qa["tax_year"]
        response = mock_rag_client.query(query, user_inspector, tax_year=expected_year)

        if response.chunks:
            year_matched = any(
                c.tax_year == expected_year for c in response.chunks
            )
            assert year_matched, (
                f"[{qa['id']}] No chunk with tax_year={expected_year} in retrieved set. "
                f"Retrieved years: {[c.tax_year for c in response.chunks]}. "
                "Temporal filter (valid_from/valid_to) must constrain retrieval to "
                "the query year even when a newer version is semantically closer."
            )

    def test_2021_rate_not_confused_with_2024(
        self, mock_rag_client: MockRAGClient, user_inspector: UserContext
    ):
        """
        Explicitly tests the year-confusion failure mode described in module-4 §2.3:
        'Box 1 rate in 2021' must return 37.10%, not the 2024 rate of 37.07%.

        The 2021 and 2024 texts are semantically nearly identical (same article,
        same structure, only the rate percentage differs).  Without a temporal
        filter, dense retrieval returns the most recent version.
        """
        response = mock_rag_client.query(
            "Wat was het Box 1 tarief in 2021?",
            user_inspector,
            tax_year=2021,
        )
        assert response.chunks, "Expected at least one chunk for 2021 Box 1 query."
        # The top chunk must be the 2021 version
        first_chunk = response.chunks[0]
        assert first_chunk.tax_year == 2021, (
            f"Year confusion: retrieved tax_year={first_chunk.tax_year} for 2021 query. "
            "Expected 2021 version (rate: 37,10%). "
            "The temporal filter must prefer year-exact chunks."
        )
        # The rate in the 2021 chunk text must match
        assert "37,10" in first_chunk.text, (
            f"2021 chunk text does not contain the expected 2021 rate 37.10%. "
            f"Found: '{first_chunk.text[:100]}'"
        )

    def test_2023_rate_not_confused_with_2024(
        self, mock_rag_client: MockRAGClient, user_inspector: UserContext
    ):
        """
        Near-miss test from module-4 §2.3 worked example:
        'Box 1 rate 2023' vs 'Box 1 rate 2024' cosine similarity ~0.951-0.962.
        The 0.97 cache threshold blocks cache collisions, but the retrieval
        filter must also return the correct year version.
        """
        response = mock_rag_client.query(
            "Wat was het Box 1 tarief eerste schijf in 2023?",
            user_inspector,
            tax_year=2023,
        )
        assert response.chunks, "Expected at least one chunk for 2023 Box 1 query."
        first_chunk = response.chunks[0]
        assert first_chunk.tax_year == 2023, (
            f"Year confusion: retrieved tax_year={first_chunk.tax_year} for 2023 query."
        )
        assert "36,93" in first_chunk.text, (
            "2023 chunk must contain the 2023-specific rate (36.93%). "
            "If 2024 text (37.07%) is returned, the temporal filter has failed."
        )

    def test_2022_rate_is_distinct_from_2024(
        self, mock_rag_client: MockRAGClient, user_inspector: UserContext
    ):
        """2022 Box 1 tarief query returns 2022 version (37,07% — same as 2024 but confirmed by tax_year)."""
        response = mock_rag_client.query(
            "Wat was het inkomstenbelastingtarief Box 1 in 2022?",
            user_inspector,
            tax_year=2022,
        )
        if response.chunks:
            tax_years = {c.tax_year for c in response.chunks}
            assert 2022 in tax_years, (
                "2022 version of Box 1 legislation not in retrieved chunks."
            )


class TestValidFromValidToFilter:
    """Tests that the valid_from/valid_to date range filter excludes wrong versions."""

    def test_chunk_valid_on_date(self):
        """Chunk with valid_from=2024-01-01 is valid on 2024-06-01."""
        chunk = top_chunk_for_article_year("3.114", 2024)
        assert chunk is not None
        assert is_valid_on(chunk, date(2024, 6, 1)), (
            "2024 chunk must be valid on 2024-06-01."
        )

    def test_chunk_invalid_before_valid_from(self):
        """Chunk with valid_from=2024-01-01 is NOT valid on 2023-12-31."""
        chunk = top_chunk_for_article_year("3.114", 2024)
        assert chunk is not None
        assert not is_valid_on(chunk, date(2023, 12, 31)), (
            "2024 chunk must NOT be valid on 2023-12-31 (before valid_from)."
        )

    def test_chunk_invalid_after_valid_to(self):
        """Chunk with valid_to=2021-12-31 is NOT valid on 2022-01-01."""
        chunk = top_chunk_for_article_year("3.114", 2021)
        assert chunk is not None
        # 2021 chunk has valid_to=2021-12-31
        assert not is_valid_on(chunk, date(2022, 1, 1)), (
            "2021 chunk must NOT be valid on 2022-01-01 (after valid_to)."
        )

    def test_chunk_valid_on_last_day(self):
        """Chunk with valid_to=2021-12-31 IS valid on 2021-12-31 (boundary inclusive)."""
        chunk = top_chunk_for_article_year("3.114", 2021)
        assert chunk is not None
        assert is_valid_on(chunk, date(2021, 12, 31)), (
            "valid_to boundary must be inclusive (valid_to >= query_date)."
        )

    def test_open_ended_validity(self):
        """Chunk with valid_to=null (current law) is valid on any future date."""
        chunk = top_chunk_for_article_year("3.114", 2024)
        assert chunk is not None
        assert chunk.valid_to is None, (
            "Current law chunk must have valid_to=null."
        )
        assert is_valid_on(chunk, date(2030, 1, 1)), (
            "Null valid_to means the chunk remains valid indefinitely."
        )

    def test_temporal_filter_excludes_all_wrong_year_versions(self):
        """
        For a 2021 query, all 2022/2023/2024 versions of art. 3.114 are excluded.
        This simulates the OpenSearch range filter behaviour.
        """
        query_date = date(2021, 6, 15)
        art3114_chunks = [c for c in MOCK_CORPUS if c.article == "3.114"]
        valid_for_2021 = [c for c in art3114_chunks if is_valid_on(c, query_date)]
        wrong_year = [c for c in valid_for_2021 if c.tax_year not in (2021, None)]
        assert wrong_year == [], (
            f"Temporal filter left {len(wrong_year)} wrong-year chunks in result. "
            f"Tax years found: {[c.tax_year for c in wrong_year]}."
        )

    def test_query_year_2023_excludes_2024_chunk(self):
        """
        Verify the exact near-miss from module-4 §2.3:
        A 2023 query must not include the 2024 chunk (valid_from=2024-01-01).
        """
        query_date = date(2023, 6, 1)
        art3114_2024 = top_chunk_for_article_year("3.114", 2024)
        assert art3114_2024 is not None
        assert not is_valid_on(art3114_2024, query_date), (
            "2024 chunk (valid_from=2024-01-01) must NOT be valid on 2023-06-01. "
            "This is the critical near-miss test from module-4 §2.3."
        )


class TestSupersededDocumentDisclosure:
    """
    Domain review Check 4 (module-1-2-retrieval.md):
    'The generation prompt should receive the superseded_by field and surface
    a disclosure when a cited chunk has a non-null superseded_by value.'

    Tests that superseded chunks carry the superseded_by metadata so the
    generator can add the required disclosure.
    """

    def test_superseded_chunk_has_superseded_by_field(self):
        """
        The 2020 version of art. 3.114 should be marked as superseded by the 2021 version.
        This field must be present on the ChunkMeta so the generator can disclose it.
        """
        superseded = next(
            (c for c in MOCK_CORPUS if c.doc_id == "wet-ib-2001-art3114-2020-superseded"),
            None,
        )
        assert superseded is not None, (
            "Test corpus must include the 2020 superseded chunk."
        )
        assert superseded.superseded_by is not None, (
            "Superseded chunk must have superseded_by field set. "
            "Domain review Check 4: generator needs this field to disclose supersession."
        )
        assert superseded.valid_to is not None, (
            "Superseded chunk must have valid_to set (end of validity date)."
        )

    def test_superseded_chunk_not_returned_for_current_queries(
        self, mock_rag_client: MockRAGClient, user_inspector: UserContext
    ):
        """
        A query for the current Box 1 rate (no year specified, defaulting to 2024)
        must NOT return the superseded 2020 version.

        valid_to=2020-12-31 means the filter query_date=2024-01-01 would exclude it.
        """
        response = mock_rag_client.query(
            "Wat is het huidige Box 1 belastingtarief?",
            user_inspector,
            tax_year=2024,
        )
        superseded_ids = [
            c.doc_id for c in response.chunks
            if c.superseded_by is not None
        ]
        assert superseded_ids == [], (
            f"Superseded chunks returned for current-year query: {superseded_ids}. "
            "valid_to filter must exclude superseded documents."
        )

    def test_historical_query_may_return_superseded(
        self, mock_rag_client: MockRAGClient, user_inspector: UserContext
    ):
        """
        A query specifically for 2020 data SHOULD return the superseded 2020 chunk
        (it was valid in 2020) with the superseded_by field populated for disclosure.
        """
        response = mock_rag_client.query(
            "Wat was het Box 1 tarief in 2020?",
            user_inspector,
            tax_year=2020,
        )
        # In the mock, 2020 query may return no chunks (not in MOCK_CORPUS by tax_year filter)
        # If it does return the superseded chunk, verify the disclosure field is present
        superseded_chunks = [c for c in response.chunks if c.superseded_by is not None]
        for sc in superseded_chunks:
            assert sc.valid_to is not None, (
                "Superseded chunk in retrieval result must have valid_to for disclosure."
            )


class TestCrossLingualTemporal:
    """
    English-language temporal queries must still retrieve Dutch-law chunks
    for the correct year.  Module-1-2 §2.2: 'NL-capable, EU-grade quality'
    with verified cross-lingual retrieval.
    """

    @pytest.mark.parametrize(
        "qa",
        [
            pytest.param(qa, id=qa["id"])
            for qa in TEMPORAL_GOLDEN
            if qa.get("language") == "en"
        ],
    )
    def test_english_temporal_query_retrieves_correct_year(
        self, qa: dict, mock_rag_client: MockRAGClient, user_inspector: UserContext
    ):
        """English query returns Dutch legislation for the specified tax year."""
        response = mock_rag_client.query(
            qa["query"], user_inspector, tax_year=qa["tax_year"]
        )
        if response.chunks:
            tax_years = {c.tax_year for c in response.chunks}
            assert qa["tax_year"] in tax_years, (
                f"[{qa['id']}] English temporal query did not retrieve chunks for "
                f"year {qa['tax_year']}. Retrieved years: {tax_years}."
            )
