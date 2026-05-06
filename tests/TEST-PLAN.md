# Test Plan — Tax Authority RAG Evaluation Suite

Stack: pytest + Ragas-style metrics + DeepEval-style red-team + Haiku 4.5 LLM-as-judge.
Bedrock model IDs match `MASTER-PLAN.md` and `module-4-ops-security.md` Appendix C.

Run modes:
- Fast loop: `pytest -m "not slow"` (default, < 2 min, no Bedrock spend on stub paths).
- Full eval: `pytest` (includes integration + latency + judge-LLM tests; runs against the docker-compose stack supplied by `docker-runner`).

## File-by-file

| File | Test count (≈) | Threat / quality dimension | Pass criterion |
|---|---|---|---|
| `conftest.py` | (fixtures) | Promotion-gate constants, role matrix, Bedrock judge, OpenSearch + Redis handles, `MockRAGClient`, `MOCK_CORPUS`, `assert_no_fiod_content`, `build_cache_key`. | n/a — provides shared scaffolding. |
| `golden/golden_qa.jsonl` | 50 pairs | Curated Q&A spanning legislation / case_law / policy / elearning × 2021-2024 × public/internal/fiod, plus 10 RBAC-must-refuse, 8 ambiguous/refusal, 8 multi-level citations (article+lid+onderdeel), 6 ECLI exact-match, 8 temporal. | Used as parameterise source by every test below. |
| `test_citation_accuracy.py` | ~10 | **Zero-hallucination citation guard**: every cited `(doc_id, article, lid, onderdeel)` tuple must exist in the retrieved chunks AND the cited text must entail the claim. Includes deterministic anchor + Haiku NLI judge. | `Citation Accuracy = 1.00` (Module 4 §4.2 hard gate). |
| `test_rbac_redteam.py` | ~12 | **RBAC adversarial**: helpdesk asks for FIOD content via direct title, semantic paraphrase, cache-poisoning replay, prompt-injection ("ignore filters"), and the new domain-review check that `structured_refusal.closest_hits` runs through `redaction_guard`. | `RBAC Leak Rate = 0.00` (hard fail; checked first in CI). |
| `test_temporal_correctness.py` | ~8 | **Tax-year correctness**: 2021/2022/2023/2024 Box-1 rate queries; retrieved law version matches query year; `valid_from`/`valid_to` respected; superseded versions surfaced only when explicitly requested. | Retrieved chunks' `tax_year` matches query year for ≥ 95% of golden pairs. |
| `test_ambiguity_refusal.py` | ~8 | **CRAG fallback**: out-of-corpus, contradictory, under-specified queries trigger Irrelevant/Ambiguous verdict; system refuses or asks; no fabricated citations on empty grounding; loop-guard caps `attempt_count ≤ 2` and `gen_retry_count ≤ 1`. | No citations when `grounded == False`; `refusal_payload.status == "insufficient_grounding"`. |
| `test_hybrid_retrieval.py` | ~8 | **Hybrid BM25 + dense**: ECLI exact-match → rank 1 (BM25 boost); semantic queries hit dense top-5; RRF (k=60) deduplicates; mock-rerank improves nDCG@5 vs RRF-only. | `nDCG@5 (rerank) ≥ nDCG@5 (RRF)`; ECLI top-1 always. |
| `test_semantic_cache.py` | ~12 | **Cache safety**: SHA-256(emb_bucket‖role‖ceiling‖year) is role-bound, year-bound, classification-bound, deterministic; near-miss pairs (Box 1 2023 vs 2024 sim ≈ 0.955) cleared by 0.97 floor; Redis round-trip & TTL ≤ 24h. | All near-miss pairs miss at threshold; role-partitioned keys distinct. |
| `test_latency_budgets.py` | ~7 | **SLA**: 100-query smoke run; p95 TTFT ≤ 1500 ms; p99 TTFT ≤ 2500 ms; p95 e2e ≤ 4000 ms; retrieval p95 ≤ 300 ms; rerank p95 ≤ 200 ms; warm cache not slower than cold. | Each percentile under documented budget. |
| `test_observability.py` | ~10 | **Span contract**: every request emits trace_id, user_role, classification_ceiling, cache_hit; generate span has Bedrock token counts; retrieve span has `filter_clause = efficient_filter`; cite_verify span has `claims_count`, `grounded`, `unsupported_count`; ContextPrecision ≥ 0.85, ContextRecall ≥ 0.90. | All required attributes present; gates cleared. |
| `pytest.ini` | n/a | Markers: `slow`, `redteam`, `latency`, `integration`, `citation`, `cache`, `temporal`, `refusal`, `observability`. Default filter `-m "not slow"`. | n/a — runner config. |

Total runnable test functions: **~75**, parametrised over 50 golden pairs → **~250 effective assertions**.

## Domain-Review Coverage

The reviewer's three highest-severity findings each map to a concrete test:

| Domain-review finding | Test file → test function |
|---|---|
| ❌ FIOD existence-disclosure via `structured_refusal.closest_hits` | `test_rbac_redteam.py::test_refusal_does_not_leak_fiod_existence` <br> `test_ambiguity_refusal.py::test_refusal_payload_shape` |
| ❌ Citation anchor depth missing `lid` / `onderdeel` | `test_citation_accuracy.py::test_anchor_pattern_captures_lid_onderdeel` <br> `test_citation_accuracy.py::test_lid_mismatch_is_unsupported` |
| ⚠️ `eli_or_ecli` single-field collisions | `test_citation_accuracy.py::ECLI_PATTERN` separate matcher; `conftest.ChunkMeta` has `eli` and `ecli` as distinct fields |
| ⚠️ Temporal scope: superseded chunks | `test_temporal_correctness.py::test_superseded_chunk_not_returned_for_current_year` |
| ⚠️ Tax-year cache-poisoning worked example | `test_semantic_cache.py::test_year_confusion_does_not_collide` |
| ✅ `lid + onderdeel` carried in metadata schema | enforced via `conftest.ChunkMeta` and `_make_chunk(...)` |

## Promotion gates encoded as constants (`conftest.py`)

| Constant | Value | Source |
|---|---|---|
| `FAITHFULNESS_THRESHOLD` | 0.95 | Module 4 §4.2 |
| `CTX_PRECISION_THRESHOLD` | 0.85 | Module 4 §4.2 |
| `CTX_RECALL_THRESHOLD` | 0.90 | Module 4 §4.2 |
| `ANSWER_RELEVANCY_THRESHOLD` | 0.90 | Module 4 §4.2 |
| `CITATION_ACCURACY_THRESHOLD` | 1.00 | Module 4 §4.2 (hard) |
| `RBAC_LEAK_RATE` | 0.00 | Module 4 §4.2 (hard, checked first) |
| `LATENCY_P95_TTFT_MS` | 1500 | Assignment + Module 4 §4.2 |
| `LATENCY_P99_TTFT_MS` | 2500 | derived |
| `LATENCY_P95_E2E_MS` | 4000 | Module 4 §4.2 |
| `RETRIEVAL_P95_MS` | 300 | Module 4 §4.2 |
| `RERANK_P95_MS` | 200 | Module 4 §4.2 |
| `CACHE_COSINE_THRESHOLD` | 0.97 | Module 4 §2.3 floor |
| `CACHE_COSINE_DEFAULT` | 0.98 | Module 4 §2.3 default |

## Hand-off to docker-runner

The runner provisions `opensearch` (Lucene k-NN, security plugin), `redis-stack`, `jaeger`, and the `app` container. AWS credentials arrive via `.env` (gitignored); `BEDROCK_LLM_ID`, `BEDROCK_EMBED_ID`, `BEDROCK_RERANK_ID`, `OPENSEARCH_URL`, `REDIS_URL`, `OTEL_EXPORTER_OTLP_ENDPOINT` are read at fixture init. The runner executes `pytest --junitxml=reports/junit.xml --html=reports/report.html` and writes `reports/test-results.md` — that artefact is the empirical evidence for the report-compiler.
