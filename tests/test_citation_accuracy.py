"""
test_citation_accuracy.py — Zero-hallucination citation guard.

Threat/quality dimension: Any cited (doc_id, article, lid, onderdeel) tuple must:
  1. Exist in the retrieved context (chunk lookup — deterministic).
  2. The cited paragraph text must actually substantiate the claim (Haiku NLI judge).

Domain review findings addressed:
  - Check 2 (Hierarchy Depth): citation match includes lid + onderdeel, not just article.
    The domain reviewer flagged that ANCHOR_PATTERN in module-3 §4.1 only captures
    (doc_id, article, paragraph) — a citation correct at article level but wrong at
    lid/onderdeel level would pass.  These tests enforce full depth.
  - Check 1 (eli_or_ecli single-field): ECLI citations in generated text are parsed
    separately via ECLI_PATTERN.

Pass criterion: Citation Accuracy = 1.00 (promotion gate, module-4 §4.2).
Every unsupported claim is a hard failure.

Stack: Ragas Faithfulness + DeepEval FaithfulnessMetric + custom Haiku NLI judge.
"""

from __future__ import annotations

import json
import re
from typing import Optional

import pytest

from conftest import (
    CITATION_ACCURACY_THRESHOLD,
    FAITHFULNESS_THRESHOLD,
    MOCK_CORPUS,
    CORPUS_BY_DOC_ID,
    ChunkMeta,
    HaikuJudge,
    MockRAGClient,
    RAGResponse,
    UserContext,
    load_golden_qa,
)

# ---------------------------------------------------------------------------
# Citation anchor patterns
# ---------------------------------------------------------------------------

# Core anchor: (doc_id=..., art. ..., lid ..., onderdeel ...)
# The full pattern captures lid and onderdeel (domain review Check 2 fix).
ANCHOR_PATTERN = re.compile(
    r"\(doc_id=(?P<doc_id>[^,)\s]+)"
    r"(?:,\s*art\.?\s*(?P<article>[^,)]+))?"
    r"(?:,\s*lid\s*(?P<lid>[^,)]+))?"
    r"(?:,\s*onderdeel\s*(?P<onderdeel>[^,)]+))?"
    r"(?:,\s*sub\s*(?P<sub>[^,)]+))?"
    r"\)",
    re.IGNORECASE,
)

# Separate ECLI pattern for case-law citations  (domain review Check 1)
ECLI_PATTERN = re.compile(
    r"ECLI:NL:[A-Z]+:\d{4}:\d+",
    re.IGNORECASE,
)


def extract_citation_tuples(answer_text: str) -> list[dict]:
    """
    Extract all citation anchors from generated answer text.
    Returns list of dicts with keys: doc_id, article, lid, onderdeel, sub.
    Includes ECLI citations as separate entries.
    """
    citations = []
    for m in ANCHOR_PATTERN.finditer(answer_text):
        citations.append({
            "doc_id":    m.group("doc_id"),
            "article":   (m.group("article") or "").strip(),
            "lid":       (m.group("lid") or "").strip(),
            "onderdeel": (m.group("onderdeel") or "").strip(),
            "sub":       (m.group("sub") or "").strip(),
        })
    for m in ECLI_PATTERN.finditer(answer_text):
        citations.append({
            "doc_id":  None,
            "ecli":    m.group(0),
            "article": None,
            "lid":     None,
            "onderdeel": None,
            "sub":     None,
        })
    return citations


def chunk_matches_citation(chunk: ChunkMeta, citation: dict) -> bool:
    """
    Step 2 of citation verifier (module-3 §4.1):
    Check whether a retrieved chunk satisfies a parsed citation tuple.

    Matching rules:
    - doc_id must match (or ecli must match chunk.ecli).
    - article must match if specified in citation (not empty).
    - lid must match if specified in citation (domain review Check 2 — not just article).
    - onderdeel must match if specified in citation.
    - sub must match if specified in citation.

    A citation that is correct at article level but wrong at lid/onderdeel level
    returns False — this is the hallucination the domain reviewer flagged.
    """
    ecli_citation = citation.get("ecli")
    if ecli_citation:
        return chunk.ecli == ecli_citation

    if citation.get("doc_id") and chunk.doc_id != citation["doc_id"]:
        return False

    if citation.get("article") and chunk.article != citation["article"]:
        return False

    # lid match — enforce if specified  (domain review Check 2)
    if citation.get("lid") and chunk.lid != citation["lid"]:
        return False

    # onderdeel match — enforce if specified  (domain review Check 2)
    if citation.get("onderdeel") and chunk.onderdeel != citation["onderdeel"]:
        return False

    # sub match — enforce if specified  (domain review Check 2, hierarchy_path sub fix)
    if citation.get("sub") and chunk.sub != citation["sub"]:
        return False

    return True


def verify_citations_deterministic(
    answer_text: str,
    retrieved_chunks: list[ChunkMeta],
) -> tuple[list[dict], list[dict]]:
    """
    Step 1 + Step 2 of citation verifier (module-3 §4.1).
    Returns (all_citations, unsupported_citations).
    unsupported = cited tuple has no matching chunk in retrieved context.
    """
    all_cits = extract_citation_tuples(answer_text)
    unsupported = []
    for cit in all_cits:
        if not any(chunk_matches_citation(c, cit) for c in retrieved_chunks):
            unsupported.append(cit)
    return all_cits, unsupported


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CITATION_GOLDEN = [
    qa for qa in load_golden_qa()
    if qa.get("expected_citations") and not qa.get("must_refuse")
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCitationAnchorDepth:
    """
    Validates that the citation parser enforces lid + onderdeel depth,
    not just (doc_id, article).

    Domain review Check 2: 'extend ANCHOR_PATTERN and Step 2 lookup to include
    lid and onderdeel fields.'
    """

    def test_anchor_extracts_full_depth(self):
        """Parser extracts article, lid, onderdeel, and sub from a citation string."""
        answer = (
            "De aftrek is mogelijk op grond van "
            "(doc_id=wet-ib-2001-art316, art. 3.16, lid 2, onderdeel a)"
        )
        cits = extract_citation_tuples(answer)
        assert len(cits) == 1
        assert cits[0]["doc_id"] == "wet-ib-2001-art316"
        assert cits[0]["article"] == "3.16"
        assert cits[0]["lid"] == "2"
        assert cits[0]["onderdeel"] == "a"

    def test_anchor_extracts_sub_level(self):
        """Parser extracts sub-level (e.g. sub 3) from a citation string."""
        answer = (
            "Zie (doc_id=wet-ib-2001-art3114-lid2-sub, art. 3.114, lid 2, onderdeel a, sub 3)"
        )
        cits = extract_citation_tuples(answer)
        assert len(cits) == 1
        assert cits[0]["sub"] == "3"

    def test_wrong_lid_is_unsupported(self):
        """
        A citation citing lid 3 when only lid 2 is retrieved must be flagged
        as unsupported — article alone is not sufficient.

        This is the exact hallucination risk the domain reviewer described:
        both chunks share article=3.114 but lid 2 vs lid 3 is material.
        """
        answer = "(doc_id=wet-ib-2001-art3114, art. 3.114, lid 3)"
        # Corpus only has lid=1 and lid=2 for this article
        chunks = [c for c in MOCK_CORPUS if c.doc_id == "wet-ib-2001-art3114"]
        all_cits, unsupported = verify_citations_deterministic(answer, chunks)
        assert len(unsupported) == 1, (
            "Citation with wrong lid should be unsupported. "
            "If this fails, lid matching is not enforced (domain review Check 2)."
        )

    def test_wrong_onderdeel_is_unsupported(self):
        """
        Citing onderdeel b when only onderdeel a is retrieved must fail.
        Leden of art. 3.16 differ in meaning — b excludes aftrek, a allows it.
        """
        answer = "(doc_id=wet-ib-2001-art316, art. 3.16, lid 2, onderdeel b)"
        # Retrieve only onderdeel a chunk
        chunks = [c for c in MOCK_CORPUS if c.doc_id == "wet-ib-2001-art316"]
        all_cits, unsupported = verify_citations_deterministic(answer, chunks)
        # wet-ib-2001-art316 has onderdeel a; the citation says b → unsupported
        assert len(unsupported) == 1, (
            "Citation with wrong onderdeel should be unsupported. "
            "If this fails, onderdeel matching is not enforced (domain review Check 2)."
        )

    def test_correct_full_depth_citation_is_supported(self):
        """A citation matching doc_id, article, lid, and onderdeel exactly passes Step 2."""
        answer = "(doc_id=wet-ib-2001-art316, art. 3.16, lid 2, onderdeel a)"
        chunks = [c for c in MOCK_CORPUS if c.doc_id == "wet-ib-2001-art316"]
        all_cits, unsupported = verify_citations_deterministic(answer, chunks)
        assert len(unsupported) == 0

    def test_sub_level_mismatch_is_unsupported(self):
        """
        A citation claiming sub 4 when only sub 3 exists must be unsupported.
        This covers the domain-reviewer flag about hierarchy_path missing sub.
        """
        answer = (
            "(doc_id=wet-ib-2001-art3114-lid2-sub, art. 3.114, lid 2, onderdeel a, sub 4)"
        )
        chunks = [c for c in MOCK_CORPUS if c.chunk_id == "wet-ib-2001-art3114-lid2-a-sub3-chunk1"]
        all_cits, unsupported = verify_citations_deterministic(answer, chunks)
        assert len(unsupported) == 1

    def test_ecli_citation_matched_by_ecli_field(self):
        """ECLI citations in text are matched against chunk.ecli, not doc_id."""
        answer = (
            "Zie het arrest ECLI:NL:HR:2023:123 voor de conclusie over reiskosten."
        )
        ecli_chunk = next(c for c in MOCK_CORPUS if c.ecli == "ECLI:NL:HR:2023:123")
        all_cits, unsupported = verify_citations_deterministic(answer, [ecli_chunk])
        assert len(unsupported) == 0

    def test_nonexistent_ecli_is_unsupported(self):
        """An ECLI in the answer text that is not in retrieved chunks is unsupported."""
        answer = "Zie ECLI:NL:HR:2099:99999 voor fictieve jurisprudentie."
        chunks = [c for c in MOCK_CORPUS if c.doc_type == "case_law"]
        all_cits, unsupported = verify_citations_deterministic(answer, chunks)
        assert len(unsupported) == 1


class TestCitationNLIJudge:
    """
    Step 3: Haiku NLI judge checks that the cited paragraph text
    actually entails the claim (module-3 §4.1, Step 3).

    These tests make real Bedrock calls — marked 'integration'.
    """

    @pytest.mark.integration
    def test_faithful_claim_is_entailed(self, haiku_judge: HaikuJudge):
        """A claim that is a faithful paraphrase of the source is 'entailed'."""
        claim = "Het Box 1 tarief bedraagt 37,07% in 2024 voor inkomens in de eerste schijf."
        source = (
            "Het belastbare inkomen uit werk en woning wordt belast tegen een tarief van "
            "37,07 procent voor zover het de ondergrens van de tweede schijf niet overschrijdt "
            "en 49,50 procent daarboven (2024)."
        )
        result = haiku_judge.judge_entailment(claim, source)
        assert result["result"] == "entailed", (
            f"Expected claim to be entailed. Explanation: {result.get('explanation')}"
        )

    @pytest.mark.integration
    def test_hallucinated_claim_is_not_entailed(self, haiku_judge: HaikuJudge):
        """A claim introducing facts not in the source is 'not_entailed'."""
        claim = "Het Box 1 tarief is 25% voor alle belastingplichtigen in 2024."
        source = (
            "Het belastbare inkomen uit werk en woning wordt belast tegen een tarief van "
            "37,07 procent voor zover het de ondergrens van de tweede schijf niet overschrijdt."
        )
        result = haiku_judge.judge_entailment(claim, source)
        assert result["result"] == "not_entailed", (
            "Hallucinated claim (wrong rate) must be rejected by NLI judge."
        )

    @pytest.mark.integration
    def test_wrong_year_claim_is_not_entailed(self, haiku_judge: HaikuJudge):
        """
        Claiming the 2024 rate for 2021 must be not_entailed against the 2021 source.
        This prevents year-confusion hallucination surviving the NLI check.
        """
        claim = "In 2021 bedroeg het Box 1 tarief 37,07 procent."
        source = (
            "Het belastbare inkomen uit werk en woning wordt belast tegen een tarief van "
            "37,10 procent voor zover het de ondergrens van de tweede schijf niet overschrijdt "
            "en 49,50 procent daarboven (2021)."
        )
        result = haiku_judge.judge_entailment(claim, source)
        assert result["result"] == "not_entailed", (
            "Wrong-year rate claim (37.07 vs 37.10) must fail NLI check."
        )

    @pytest.mark.integration
    def test_faithfulness_score_passes_threshold(self, haiku_judge: HaikuJudge):
        """End-to-end faithfulness score from Haiku must meet the 0.95 promotion gate."""
        answer = (
            "Het Box 1 tarief voor 2024 bedraagt 37,07 procent in de eerste schijf "
            "en 49,50 procent daarboven."
        )
        context_chunks = [
            "Het belastbare inkomen uit werk en woning wordt belast tegen een tarief van "
            "37,07 procent voor zover het de ondergrens van de tweede schijf niet overschrijdt "
            "en 49,50 procent daarboven (2024)."
        ]
        score = haiku_judge.judge_faithfulness(answer, context_chunks)
        assert score >= FAITHFULNESS_THRESHOLD, (
            f"Faithfulness score {score:.3f} below threshold {FAITHFULNESS_THRESHOLD}. "
            "Promotion gate: Faithfulness >= 0.95 (module-4 §4.2)."
        )

    @pytest.mark.integration
    def test_faithfulness_score_fails_on_hallucination(self, haiku_judge: HaikuJudge):
        """Answer with fabricated data must score below the faithfulness threshold."""
        answer = (
            "Het Box 1 tarief voor 2024 bedraagt 22 procent. "
            "Dit is vastgelegd in artikel 99 van de Wet IB."
        )
        context_chunks = [
            "Het belastbare inkomen uit werk en woning wordt belast tegen een tarief van "
            "37,07 procent voor zover het de ondergrens van de tweede schijf niet overschrijdt."
        ]
        score = haiku_judge.judge_faithfulness(answer, context_chunks)
        assert score < FAITHFULNESS_THRESHOLD, (
            f"Hallucinated answer scored {score:.3f} >= {FAITHFULNESS_THRESHOLD}. "
            "The judge should have flagged the fabricated rate."
        )


class TestCitationOnGoldenSet:
    """
    Parametrized citation-accuracy tests over the golden Q&A set.
    Each item in the golden set with expected_citations is exercised.
    """

    @pytest.mark.parametrize(
        "qa",
        [
            pytest.param(qa, id=qa["id"])
            for qa in CITATION_GOLDEN
        ],
    )
    def test_mock_response_citations_are_grounded(
        self, qa: dict, mock_rag_client: MockRAGClient, user_inspector: UserContext
    ):
        """
        For each golden Q&A pair, the mock RAG client retrieves relevant chunks
        and generates a deterministic answer.  Every citation in the answer must
        be traceable back to a retrieved chunk.

        This validates the citation verifier logic (module-3 §4.1, Steps 1+2)
        independent of the LLM.  Real Bedrock calls are reserved for NLI judge tests.
        """
        query = qa["query"]
        tax_year = qa.get("tax_year")
        response: RAGResponse = mock_rag_client.query(
            query, user_inspector, tax_year=tax_year
        )

        if response.answer is None:
            # Refusal is acceptable for empty-corpus queries
            return

        all_cits, unsupported = verify_citations_deterministic(
            response.answer, response.chunks
        )

        assert len(unsupported) == 0, (
            f"[{qa['id']}] Unsupported citations found: {unsupported}. "
            "Citation Accuracy must = 1.00 (module-4 §4.2 hard fail)."
        )

    @pytest.mark.parametrize(
        "qa",
        [
            pytest.param(qa, id=qa["id"])
            for qa in CITATION_GOLDEN
            if qa.get("expected_citations")
        ],
    )
    def test_expected_citations_match_lid_onderdeel(self, qa: dict):
        """
        For golden items with expected_citations that specify lid and/or onderdeel,
        verify that the corresponding chunk exists in MOCK_CORPUS with matching depth.

        This is a corpus-integrity test: if the corpus is missing a lid/onderdeel
        combination required by a golden item, the test fails loudly rather than
        silently passing with a shallow citation.
        """
        for expected_cit in qa.get("expected_citations", []):
            doc_id = expected_cit.get("doc_id")
            if not doc_id:
                continue
            chunks = CORPUS_BY_DOC_ID.get(doc_id, [])
            assert chunks, (
                f"[{qa['id']}] Expected doc_id '{doc_id}' not found in MOCK_CORPUS."
            )
            if expected_cit.get("lid"):
                matching = [c for c in chunks if c.lid == expected_cit["lid"]]
                assert matching, (
                    f"[{qa['id']}] No chunk with doc_id='{doc_id}' and "
                    f"lid='{expected_cit['lid']}' in MOCK_CORPUS. "
                    "Domain review Check 2: corpus must have lid-level granularity."
                )
            if expected_cit.get("onderdeel"):
                matching = [
                    c for c in chunks
                    if c.lid == expected_cit.get("lid")
                    and c.onderdeel == expected_cit["onderdeel"]
                ]
                assert matching, (
                    f"[{qa['id']}] No chunk with doc_id='{doc_id}', "
                    f"lid='{expected_cit.get('lid')}', "
                    f"onderdeel='{expected_cit['onderdeel']}' in MOCK_CORPUS."
                )


class TestCitationOnEmptyGrounding:
    """
    When grounding is empty (CRAG Irrelevant path), the answer must contain
    NO citation anchors — the refusal payload must not fabricate any.
    Module-3 §2.2 structured_refusal: 'Never calls a generator. Zero hallucination risk.'
    """

    def test_structured_refusal_has_no_citations(
        self, mock_rag_client: MockRAGClient, user_helpdesk: UserContext
    ):
        """
        Query with no corpus match → refusal response must contain zero citation anchors.
        """
        response = mock_rag_client.query(
            "Uitleg over niet-bestaand fiscaal concept XYZ-alpha-99",
            user_helpdesk,
        )
        assert response.answer is None, "Empty-corpus query must produce a refusal."
        assert response.refusal_payload is not None
        assert response.citations == [], (
            "Refusal response must have empty citations list."
        )

    def test_ambiguous_response_has_no_fabricated_article(
        self, mock_rag_client: MockRAGClient, user_helpdesk: UserContext
    ):
        """
        Ambiguous query with no corpus match must not contain a fabricated article number.
        Checks that 'Artikel 99.999' or any non-corpus article does not appear.
        """
        response = mock_rag_client.query(
            "Artikel 99.999 van de Wet IB 2001",
            user_helpdesk,
        )
        # If answer is returned, it must not cite the non-existent article
        if response.answer:
            cits = extract_citation_tuples(response.answer)
            fabricated = [c for c in cits if c.get("article") == "99.999"]
            assert not fabricated, (
                "Answer must not cite a fabricated article number (99.999)."
            )
