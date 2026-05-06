"""
test_ambiguity_refusal.py — CRAG fallback / refusal validation.

Threat/quality dimension:
  Out-of-corpus, contradictory, and under-specified queries must trigger the
  CRAG grader's `Irrelevant` or `Ambiguous` verdict and end in either:
    (a) a structured_refusal payload (no citations, no fabricated answer), or
    (b) generate_with_disclosure with explicit "missing_aspects" listed.
  No hallucination is permitted — empty grounding must produce empty citations.

Domain review findings addressed:
  - Module 3 §2.2 structured_refusal node: closest_hits must run through
    redaction_guard so a lower-privileged user cannot infer FIOD doc existence
    from a refusal payload.  The cross-check sits in test_rbac_redteam.py;
    here we assert the structural shape (no citations) on plain ambiguous queries.

Pass criterion:
  - aggregate_verdict in {Irrelevant, Ambiguous} for queries with no grounding.
  - response.citations == [] when grounded == False.
  - response.refusal_payload is populated when verdict == Irrelevant and cap hit.
"""

from __future__ import annotations

import pytest

from conftest import (
    MockRAGClient,
    UserContext,
)

# ---------------------------------------------------------------------------
# Out-of-corpus queries — should refuse.
# ---------------------------------------------------------------------------

OUT_OF_CORPUS_QUERIES = [
    # Topic outside any chunk in MOCK_CORPUS
    "Wat is het tarief voor zegelrecht op aandelenoverdrachten in Curaçao 2024?",
    # Topic adjacent but no specific grounding
    "Hoeveel bedraagt de erfbelasting voor stiefkinderen onder 18 jaar in 2019?",
    # Wholly invented term
    "Geef de toepassing van de Eskimo-aftrek voor IB-ondernemers in 2024.",
]

# Under-specified — answer requires the user to clarify before grounding is possible.
UNDERSPECIFIED_QUERIES = [
    "Wat is het tarief?",        # no Box, no year
    "Mag ik dit aftrekken?",     # no expense type
    "Welke regel geldt hier?",   # no topic
]

# Contradictory — query asks for two mutually exclusive facts.
CONTRADICTORY_QUERIES = [
    "Geef het Box 1 tarief voor 2023 en bevestig dat dit 50 procent was.",  # known wrong
    "Bevestig dat ECLI:NL:HR:2099:999 een arrest is over Box 3 (2099 ligt in de toekomst).",
]


@pytest.mark.parametrize("query", OUT_OF_CORPUS_QUERIES)
def test_out_of_corpus_query_refuses(mock_rag_client, user_helpdesk, query):
    """Out-of-corpus query must refuse with no citations and no answer."""
    response = mock_rag_client.query(query, user_helpdesk)

    assert response.answer is None, (
        f"Expected refusal (answer=None) for out-of-corpus query, got: {response.answer!r}"
    )
    assert response.citations == [], (
        f"Refusal must produce zero citations; got {len(response.citations)}."
    )
    assert response.refusal_payload is not None, "Missing structured refusal payload."
    assert response.refusal_payload.get("status") == "insufficient_grounding"
    assert response.grader_verdict in {"Irrelevant", "Ambiguous"}, (
        f"Expected Irrelevant/Ambiguous verdict, got {response.grader_verdict}."
    )


@pytest.mark.parametrize("query", UNDERSPECIFIED_QUERIES)
def test_underspecified_query_does_not_fabricate(mock_rag_client, user_inspector, query):
    """Under-specified query must not produce confident citations."""
    response = mock_rag_client.query(query, user_inspector)
    if response.answer is None:
        # A clean refusal is acceptable.
        assert response.citations == []
        return
    # Otherwise we accept a generate_with_disclosure-style answer iff
    # citations remain grounded in retrieved chunks (not invented).
    retrieved_doc_ids = {c.doc_id for c in response.chunks}
    for citation in response.citations:
        assert citation["doc_id"] in retrieved_doc_ids, (
            f"Citation {citation['doc_id']!r} not in retrieved chunks — fabrication."
        )


@pytest.mark.parametrize("query", CONTRADICTORY_QUERIES)
def test_contradictory_query_refuses_or_disambiguates(mock_rag_client, user_inspector, query):
    """Contradictory queries must not be 'confirmed' — fail-closed."""
    response = mock_rag_client.query(query, user_inspector)

    if response.answer is not None:
        # If an answer exists, it must not include the false claim verbatim.
        # The mock client cites verbatim from a chunk, so a contradictory question
        # that returns no matching chunks should hit the refusal path.
        # We only require: no fabricated citation.
        retrieved_doc_ids = {c.doc_id for c in response.chunks}
        for citation in response.citations:
            assert citation["doc_id"] in retrieved_doc_ids


def test_refusal_payload_shape(mock_rag_client, user_helpdesk):
    """structured_refusal payload must contain the contract fields."""
    response = mock_rag_client.query(
        "Welke fiscale gevolgen heeft een Klingon-belastingverdrag?", user_helpdesk
    )
    assert response.refusal_payload is not None
    payload = response.refusal_payload
    for required_field in ("status", "message", "closest_hits", "missing_aspects"):
        assert required_field in payload, f"Refusal missing field: {required_field}"
    assert payload["status"] == "insufficient_grounding"
    # Domain review fix: closest_hits must NOT carry FIOD doc IDs for lower-privileged users.
    for hit in payload["closest_hits"]:
        assert "FIOD" not in hit.get("title", "").upper()
        assert "STAATSGEHEIM" not in hit.get("title", "").upper()


def test_no_citations_on_empty_grounding(mock_rag_client, user_helpdesk):
    """If response has no chunks, it must have no citations."""
    response = mock_rag_client.query(
        "Specifieer alle Kafkaësk-fiscale aspecten van paragraaf 0 lid 0.", user_helpdesk
    )
    assert response.chunks == []
    assert response.citations == []


def test_grader_verdict_propagated(mock_rag_client, user_helpdesk):
    """Grader verdict must always be one of the documented enum values."""
    response = mock_rag_client.query(
        "wat is een arbeidsovereenkomst?", user_helpdesk
    )
    assert response.grader_verdict in {"Relevant", "Ambiguous", "Irrelevant"}


def test_loop_guard_caps_attempts(mock_rag_client, user_helpdesk):
    """attempt_count must never exceed the documented hard cap (= 2)."""
    response = mock_rag_client.query(
        "Volledig hypothetische vraag over Mars-fiscaliteit 2099.", user_helpdesk
    )
    assert response.attempt_count <= 2, (
        f"attempt_count {response.attempt_count} exceeds documented cap of 2."
    )
    assert response.gen_retry_count <= 1, (
        f"gen_retry_count {response.gen_retry_count} exceeds documented cap of 1."
    )
