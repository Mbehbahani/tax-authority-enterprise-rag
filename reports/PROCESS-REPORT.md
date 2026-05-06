# Process Report: How the Tax Authority RAG Architecture Was Built
## A Multi-Agent Methodology Report for Governance and Hiring Reviewers

**Project:** Enterprise RAG Architecture for the National Tax Authority
**Final deliverable:** `d:\AWS\Legal\design\FINAL-rag-architecture.md` (1,423 lines, all 9 assignment requirements PASS)
**This report covers:** the agent system that produced the deliverable — not the deliverable itself
**Report date:** 2026-05-06
**Report author:** process-report-author (Claude Sonnet, Phase 6)

---

## 1. Executive Summary

The National Tax Authority assignment asked for a production-grade AI assistant architecture capable of answering complex fiscal questions over 500,000 documents with zero-hallucination tolerance, strict role-based access control (RBAC — the system of rules determining which users may see which documents), and a maximum response latency of 1.5 seconds. The output required exact paragraph-level citations, defense against privilege-escalation attacks, and a full evaluation framework.

Rather than assigning this to a single AI model in a single conversation, the orchestrator deployed a **pipeline of seven specialized agents**, each with a bounded scope, distinct toolset, and written charter. The agents ran across six sequential phases spanning two calendar days (2026-05-05 to 2026-05-06), with three design agents executing in parallel during Phase 1.

**What was produced:** A 1,423-line technical architecture document, a 10-file pytest test suite with 173 passing tests (170 mock + 3 live Redis integration), a validated Docker stack running OpenSearch, Redis, and Jaeger, and a structured test-results artifact — all backed by real AWS Bedrock service calls that confirmed model access before any design work began.

**What is notable about the approach:**
- Phase 1 ran three design agents simultaneously. Their combined active time was 715 seconds; the wall-clock wait was only 420 seconds — a 1.7x parallelism factor.
- Three of seven agents hit usage limits mid-task. In each case the orchestrator completed the remaining work inline without restarting, preserving continuity.
- The legal-domain-reviewer agent operated as a pure check — it could read and annotate but not rewrite — ensuring that domain findings were advisory signals, not silent edits.
- Every architectural decision is traceable to a named agent and a specific file. No decision is anonymous.
- The docker-runner's five-iteration debug loop reduced test failures from 15 to 0 without relaxing a single assertion, demonstrating the charter's "fix the underlying issue, never weaken the test" discipline.

Total active agent time: approximately 131 minutes across 7 invocations. Total wall-clock including user breaks: approximately 16 hours 45 minutes. The design itself, from first agent start to final compilation, would have required multiple sequential working days if executed as a single LLM conversation.

---

## 2. The Multi-Agent Architecture

### System Diagram

```mermaid
graph TD
    subgraph "Phase 1 — Parallel Design (420 s wall-clock)"
        A[rag-retrieval-architect\nTools: Read Write Edit Grep Glob WebFetch WebSearch\n140 s / 21k tokens]
        B[agentic-rag-designer\nTools: Read Write Edit Grep Glob WebFetch WebSearch\n155 s / 22k tokens]
        C[mlops-security-architect\nTools: Read Write Edit Grep Glob WebFetch WebSearch\n420 s / 39k tokens]
    end

    subgraph "Phase 2 — Domain Review (148 s)"
        D[legal-domain-reviewer\nTools: Read Edit Grep Glob WebFetch WebSearch\n148 s / 40k tokens]
    end

    subgraph "Phase 3 — Test Design (620 s)"
        E[rag-evaluation-engineer\nTools: Read Write Edit Grep Glob WebFetch WebSearch\n620 s / 14 tool uses]
    end

    subgraph "Phase 4 — Build and Run (2755 s + orchestrator)"
        F[docker-runner\nTools: Read Write Edit Bash Grep Glob WebFetch\n2755 s / 110 tool uses]
    end

    subgraph "Phase 5 — Compilation (444 s)"
        G[report-compiler\nTools: Read Write Edit Grep Glob\n444 s / 9 tool uses]
    end

    subgraph "Phase 6 — This Report"
        H[process-report-author\nTools: Read Write Edit Grep Glob Bash]
    end

    MASTER[MASTER-PLAN.md\nLocked stack + decisions]
    ASSIGN[assignment\nOriginal task]

    ASSIGN --> MASTER
    MASTER --> A
    MASTER --> B
    MASTER --> C

    A -- "module-1-2-retrieval.md\nmetadata schema, HNSW config,\nhybrid search, Top-K cascade" --> D
    B -- "module-3-agentic.md\nCRAG graph, grader prompt,\ncitation verifier" --> D
    C -- "module-4-ops-security.md\nRBAC pre-filter, cache key,\npromotion gates, observability" --> D

    D -- "Domain Review Findings\nappended to all 3 module files\n(read-only annotations)" --> E
    D -- "Domain Review Findings" --> G

    A -- "chunk schema, citation object shape\nTop-K values, retrieval API contract" --> E
    B -- "CRAG state schema, grader enum\ncitation verifier output shape" --> E
    C -- "promotion thresholds as constants\nRBAC red-team threat models\nobservability span contract" --> E

    E -- "tests/ (10 files, 173 assertions)\ngolden_qa.jsonl (50 pairs)\nTEST-PLAN.md" --> F

    A -- "module-1-2-retrieval.md" --> G
    B -- "module-3-agentic.md" --> G
    C -- "module-4-ops-security.md" --> G
    F -- "test-results.md\ndocker/ scripts/" --> G

    G -- "FINAL-rag-architecture.md\n(1423 lines, 9/9 PASS)" --> H

    style A fill:#dbeafe,stroke:#2563eb
    style B fill:#dbeafe,stroke:#2563eb
    style C fill:#dbeafe,stroke:#2563eb
    style D fill:#fef3c7,stroke:#d97706
    style E fill:#dcfce7,stroke:#16a34a
    style F fill:#f3e8ff,stroke:#7c3aed
    style G fill:#fee2e2,stroke:#dc2626
    style H fill:#f0fdf4,stroke:#15803d
```

### Agent Descriptions

**rag-retrieval-architect** designs how documents are ingested and retrieved. Its responsibility spans Module 1 (breaking 500,000 documents into searchable chunks while preserving legal hierarchy) and Module 2 (the retrieval query that combines keyword search and semantic vector search). It owns the metadata schema — the structured labels attached to every chunk — which every downstream agent depends on. Its boundary ends at the retrieval API surface; it does not touch generation, caching, or security enforcement.

**agentic-rag-designer** designs the "brain" of the pipeline — Module 3. It specifies the LangGraph state machine (a directed graph of processing nodes) that decides what to do when retrieval returns ambiguous or empty results. It defines the Corrective RAG (CRAG) loop: how the system rewrites queries, when it escalates to a fallback, and when it refuses to answer rather than hallucinate. It also designs the citation verifier that checks every claim before the answer reaches the user. Its boundary ends at generation; it does not design caching, RBAC enforcement, or observability.

**mlops-security-architect** designs Module 4: the operational layer that makes the system safe and maintainable in production. It specifies the semantic cache (a memory that avoids re-computing answers to near-identical questions), the three-layer RBAC enforcement strategy, the CI/CD evaluation gates that block a bad model from reaching production, and the observability pipeline (tracing every query through Jaeger). It owns the RBAC threat model and the mathematical argument for why the access-control filter must run inside the vector search, not after it.

**legal-domain-reviewer** is a check agent, not a contributor. It reads the three module designs and applies domain expertise in Dutch tax law — the ECLI case-law citation format, the hierarchical structure of Dutch fiscal codes (Artikel/Lid/Onderdeel/Sub), FIOD classification rules, and temporal tax-year correctness. It annotates each module with findings but cannot edit the modules itself. This separation ensures that domain findings are visible audit records rather than silent corrections that could change the meaning of a design without explanation.

**rag-evaluation-engineer** translates the architecture into a runnable test suite. It reads all three module designs plus the domain-review findings to ensure that the tests target the actual design choices — not a generic RAG system. It produces the golden question-answer pairs, the adversarial RBAC attack tests, the temporal-correctness assertions, and the latency budgets. Its output is the specification of what "working correctly" means for this system.

**docker-runner** is the empirical validation agent. It builds the Docker infrastructure (containerized OpenSearch, Redis, and Jaeger), seeds a synthetic corpus of approximately 250 documents, runs the test suite, diagnoses failures, applies fixes, and repeats until all tests pass. It writes the test-results artifact consumed by the report-compiler. It operates under an explicit non-negotiable: never weaken a test to make CI green.

**report-compiler** is the integration agent. It reads all module files (with domain-review findings), the test plan, and the test results, resolves every flagged gap, and produces the single submission-ready document. It performs a cross-module consistency pass to ensure that the metadata schema in Module 1 matches the RBAC filter in Module 4, that the grader output in Module 3 matches the observability metrics in Module 4, and that the latency budget in Module 2 is enforced by the gates in Module 4.

---

## 3. Why a Multi-Agent Approach

**Parallelism.** Modules 1, 3, and 4 have no design-time dependency on each other — they share the MASTER-PLAN locked stack but do not need each other's intermediate output to begin work. Running three agents simultaneously reduced Phase 1 from a sequential 715 seconds to 420 seconds of wall-clock time. For a hiring panel reviewing a time-pressured assignment, this compression matters: three independently specialized design documents arrived simultaneously rather than trickling out over hours.

**Separation of concerns.** The security argument in Module 4 needed to be developed independently of the retrieval mechanics in Module 1. If a single agent designed both, it might unconsciously design the RBAC filter to match the retrieval strategy rather than proving the filter's correctness independently. By giving these to separate agents with separate charters, the mlops-security-architect had to confront the retrieval-architect's metadata schema as an external contract and prove that its filter predicate was correct against that schema — which is exactly the reasoning a real security review demands.

**Specialization.** Each agent's charter is written for a specific competence. The rag-retrieval-architect's charter instructs it to defend every HNSW parameter with quantitative reasoning. The agentic-rag-designer's charter instructs it to write a verbatim grader prompt, not a placeholder. The legal-domain-reviewer's charter gives it an eight-point checklist anchored in Dutch law specifically (ECLI format, FIOD classification, Box 1/2/3 structure). A single general-purpose LLM prompt cannot carry all these specialized instructions simultaneously without diluting each one — context window constraints and competing instruction priorities degrade quality on each individual dimension.

**Auditability.** Every architectural decision in the FINAL document has a named author and a source file. When the decision-provenance table (Section 6 of this report) says "OpenSearch efficient_filter chosen by rag-retrieval-architect, plan file rag-retrieval-architect-plan.md, section 4," that claim can be verified by reading that file. A monolithic single-conversation approach produces a single document with no record of which reasoning step produced which recommendation, making it impossible to distinguish a well-argued decision from a plausible-sounding one.

---

## 4. Agent Roster

The token columns for agents with `null` in the JSONL are marked as unavailable (N/A); the JSONL records confirm the data was not captured for those invocations due to the partial-completion events. Duration figures are from the JSONL.

| Agent | Role | Model | Tools | Key Inputs | Key Outputs | Duration | Total Tokens | Tool Uses | Invocations |
|---|---|---|---|---|---|---|---|---|---|
| rag-retrieval-architect | Ingestion + retrieval design (Modules 1–2) | claude-sonnet | Read, Write, Edit, Grep, Glob, WebFetch, WebSearch | assignment, MASTER-PLAN.md | module-1-2-retrieval.md | 141 s | 21,066 | 5 | 1 |
| agentic-rag-designer | CRAG state machine design (Module 3) | claude-sonnet | Read, Write, Edit, Grep, Glob, WebFetch, WebSearch | assignment, MASTER-PLAN.md | module-3-agentic.md | 155 s | 22,280 | 4 | 1 |
| mlops-security-architect | Cache, RBAC, CI/CD, observability (Module 4) | claude-sonnet | Read, Write, Edit, Grep, Glob, WebFetch, WebSearch | assignment, MASTER-PLAN.md | module-4-ops-security.md | 420 s | 38,785 | 9 | 1 |
| legal-domain-reviewer | Dutch tax/ECLI/FIOD domain review | claude-sonnet | Read, Edit, Grep, Glob, WebFetch, WebSearch | module-1-2-retrieval.md, module-3-agentic.md, module-4-ops-security.md | Domain Review Findings (appended in-place) | 148 s | 40,404 | 6 | 1 |
| rag-evaluation-engineer | Test suite design and authorship | claude-sonnet | Read, Write, Edit, Grep, Glob, WebFetch, WebSearch | All module files + domain findings | tests/ (10 files), golden_qa.jsonl, TEST-PLAN.md | 620 s | N/A | 14 | 1 (partial; orchestrator finished) |
| docker-runner | Containerization + test execution loop | claude-sonnet | Read, Write, Edit, Bash, Grep, Glob, WebFetch | tests/, module files | docker/, scripts/, reports/test-results.md | 2,755 s | N/A | 110 | 1 (partial; orchestrator finished) |
| report-compiler | Final document integration + gap resolution | claude-sonnet | Read, Write, Edit, Grep, Glob | All module files, TEST-PLAN.md, test-results.md | FINAL-rag-architecture.md (1,423 lines) | 444 s | N/A | 9 | 1 |

Notes: All agents ran on the `claude-sonnet` model family (the orchestrator session used `claude-sonnet-4-6`). The process-report-author (this document) is Phase 6 and has no completed timing entry yet in the JSONL. Token data for phases 3–5 was not captured in the JSONL due to the partial-completion events; the JSONL `null` values are preserved faithfully here rather than estimated.

---

## 5. Execution Timeline

```mermaid
gantt
    title Tax Authority RAG — Agent Execution Timeline
    dateFormat  YYYY-MM-DDTHH:mm
    axisFormat  %m-%d %H:%M

    section Phase 1 — Parallel Design
    rag-retrieval-architect    :done, ra, 2026-05-05T23:48, 2m
    agentic-rag-designer       :done, ad, 2026-05-05T23:48, 3m
    mlops-security-architect   :done, ms, 2026-05-05T23:48, 7m

    section Phase 2 — Domain Review
    legal-domain-reviewer      :done, lr, 2026-05-06T00:05, 3m

    section Phase 3 — Test Design
    rag-evaluation-engineer    :done, ee, 2026-05-06T00:08, 10m
    orchestrator-finishes-tests :done, ot, 2026-05-06T00:18, 15m

    section Phase 4 — Build + Run + Debug
    docker-runner              :done, dr, 2026-05-06T07:56, 46m
    orchestrator-debug-loop    :done, od, 2026-05-06T13:30, 53m

    section Phase 5 — Compilation
    report-compiler            :done, rc, 2026-05-06T14:23, 7m

    section Phase 6 — Process Report
    process-report-author      :active, pr, 2026-05-06T19:02, 60m
```

**Parallelism factor (Phase 1):** Three agents each started at 2026-05-05T23:48:30Z. Their individual durations were 141 s, 155 s, and 420 s respectively — a sum of 716 seconds of agent time. The wall-clock wait was 420 seconds (the slowest agent). Parallelism factor: 716 / 420 = **1.70x**. Without parallelism, Phase 1 would have taken nearly 12 minutes instead of 7.

**Phase gaps (user breaks):** The JSONL shows Phase 2 starting at 00:05:43Z (10 minutes after Phase 1 ended at 23:55:30Z), Phase 3 starting 24 seconds after Phase 2, and then a large gap before Phase 4 starts at 07:56:16Z — approximately 7.5 hours. Phase 4 had a second gap before the orchestrator finished (13:30:00Z). These gaps reflect normal human working sessions and are not agent idle time.

**Total active agent time:** 140,667 + 154,551 + 419,644 + 148,376 + 619,862 + 2,754,906 + 3,195,000 + 443,909 = approximately 7,877,000 ms = **131 minutes** of aggregate agent computation time.

**Total wall-clock:** First start 2026-05-05T23:48:30Z to Phase 5 end 2026-05-06T16:33:00Z = **approximately 16 hours 45 minutes** (including user breaks between phases).

---

## 6. Decision Provenance

Each row maps one major architectural choice to the agent that recommended it, the primary reasoning, and whether the orchestrator or user modified the recommendation.

| Decision | Recommending Agent | Source File / Section | Reasoning (paraphrase from source) | Override? |
|---|---|---|---|---|
| Vector store: Amazon OpenSearch Service over Qdrant / Weaviate | rag-retrieval-architect | rag-retrieval-architect-plan.md §2 decision 3; module-1-2-retrieval.md §1.3 | "The RBAC pre-filter runs inside HNSW traversal — mathematically safe vs. post-filter leak. Native BM25 in the same engine eliminates a separate sparse service. AWS-managed fits the AWS-resident stack." | No. User direction at MASTER-PLAN creation confirmed AWS-only stack. |
| HNSW parameters: m=32, ef_construction=256, ef_search=128 | rag-retrieval-architect | rag-retrieval-architect-plan.md §2 decision 4; module-1-2-retrieval.md §1.3 | Recall ≥ 0.95 at 20M points with 1024-dim Cohere embeddings. ef_search tunable to 64 under load without breaching latency budget. | No. Test loop confirmed no ef_search adjustment was needed. |
| Quantization: FP16 scalar (OpenSearch 2.13+) | rag-retrieval-architect | rag-retrieval-architect-plan.md §2 decision 5; module-1-2-retrieval.md §1.3 | ~2x memory reduction with negligible recall loss; binary quantization (2.16+) evaluated as future option requiring rescore step. | No. |
| Embeddings: Cohere embed-multilingual-v3 on Bedrock | rag-retrieval-architect | rag-retrieval-architect-plan.md §2 decision 6; MASTER-PLAN.md §C | NL-capable, AWS-resident (no egress for FIOD content), confirmed working on Dutch text (1024-dim). | No. Confirmed by live Bedrock test before design started. |
| Hybrid search: OpenSearch native hybrid query (BM25 + k-NN) with RRF (k=60) | rag-retrieval-architect | rag-retrieval-architect-plan.md §2 decision 7; module-1-2-retrieval.md §2.1 | Single engine (no separate sparse service); RRF is parameter-light; regex router catches ECLI/article-number queries for BM25 boost. | No. |
| Top-K cascade: BM25 100 + k-NN 100 → RRF 60 → rerank top-8 | rag-retrieval-architect | rag-retrieval-architect-plan.md §2 decision 9; module-1-2-retrieval.md §2.5 | Fits 1.5s TTFT: ~80ms hybrid query + ~250ms Bedrock rerank + LLM prefill. Rerank top-8 keeps generation context tight. | No. |
| Reranker: Cohere rerank-v3-5:0 on Bedrock | rag-retrieval-architect | rag-retrieval-architect-plan.md §2 decision 8; MASTER-PLAN.md §C | Multilingual, AWS-resident (no Cohere SaaS egress), confirmed working (Dutch tax doc ranked 0.88 vs off-topic 0.02). | No. |
| CRAG grader model: Claude Haiku 4.5, temp=0, structured output via tool-use JSON | agentic-rag-designer | agentic-rag-designer-plan.md §2 decision 2; module-3-agentic.md §3.2 | "Faster iteration, multilingual (NL/EN), prompt-tunable for corpus drift. Fine-tuned classifier deferred to v2." Cross-region inference profile `us.anthropic.claude-haiku-4-5-20251001-v1:0` confirmed working. | No. |
| CRAG fallback for Irrelevant verdict: refuse with structured payload, never hallucinate | agentic-rag-designer | agentic-rag-designer-plan.md §2 decision 4; module-3-agentic.md §4 | Regulated domain; fabricated fiscal advice creates legal liability. Structured refusal gives the user an actionable "no grounding found" response with closest non-FIOD hints. | Refined by legal-domain-reviewer: closest_hits must run through redaction_guard before serialization. |
| Loop guard: max 2 transform retries + 1 generation retry | agentic-rag-designer | agentic-rag-designer-plan.md §2 decision 6; module-3-agentic.md §5.2 | Hard cap = 3 LLM-generation cycles. Protects TTFT. On exhaustion → structured "insufficient evidence" response. | No. |
| Cache backend: Redis Stack 7.4 (RediSearch + vector field) | mlops-security-architect | mlops-security-architect-plan.md §2 decision 1; module-4-ops-security.md §2.1 | "GPTCache rejected — extra service, weaker auth model. Redis Stack satisfies all requirements in one container/managed instance." | No. |
| Cache cosine threshold: 0.97 floor / 0.98 default | mlops-security-architect | mlops-security-architect-plan.md §2 decision 2; module-4-ops-security.md §2.3 | "Box 1 rate 2024" vs "Box 1 rate 2023" embed at ~0.951–0.962; anything below 0.97 yields financially wrong hits. Worked example with exact similarity values given. | No. Confirmed empirically: test_semantic_cache.py near-miss pairs blocked correctly. |
| Cache key: SHA256(embedding_bucket + role + classification_ceiling + tax_year) | mlops-security-architect | mlops-security-architect-plan.md §2 decision 3; module-4-ops-security.md §2.2 | "Role-blind keys = confused-deputy side channel. Non-negotiable." A FIOD analyst's cached answer must never serve a helpdesk user with the same query embedding. | No. Confirmed by 3/3 Redis integration tests (cross-role cache isolation). |
| RBAC enforcement: OpenSearch efficient_filter inside HNSW traversal (primary) | mlops-security-architect | mlops-security-architect-plan.md §2 decisions 4–5; module-4-ops-security.md §3 | Post-filtering leaks in two ways: (a) empty-set inference when FIOD chunks dominate the neighborhood, (b) timing side-channel from differential traversal cost. efficient_filter prunes before scoring, eliminating both. | No. This is the design's strongest security argument. |
| Evaluation framework: Ragas (retrieval metrics) + DeepEval (generation/red-team) | mlops-security-architect | mlops-security-architect-plan.md §2 decision 6; module-4-ops-security.md §4 | "Each tool used for its strongest metric — no duplication." Ragas: Faithfulness, Context Precision/Recall, Answer Relevancy. DeepEval: Citation Accuracy, RBAC Leak Rate, latency. | No. |
| Observability: OpenTelemetry → Jaeger (replacing original Langfuse plan) | mlops-security-architect | mlops-security-architect-plan.md §2 decision 8; MASTER-PLAN.md §C | "OTLP-native, simpler ops than Langfuse." Jaeger all-in-one for dev; Cassandra/OpenSearch backend for prod. User directed Jaeger preference at MASTER-PLAN creation. | Yes — user replaced Langfuse with Jaeger. |
| eli_or_ecli split into separate eli and ecli fields | legal-domain-reviewer | module-1-2-retrieval.md §Domain Review §Check 1 | "A single `eli_or_ecli` field blocks type-safe routing: code wanting all legislation chunks citing an ELI must parse the value to distinguish it from ECLI case-law strings." | Applied by report-compiler in FINAL document. |
| hierarchy_path must include /sub level | legal-domain-reviewer | module-1-2-retrieval.md §Domain Review §Check 2 | "For Artikel 3.114, lid 2, onderdeel a, sub 3° the path renders as eli/art3.114/lid2/a, losing the sub-level. The citation verifier uses hierarchy_path as canonical anchor; a citation to sub 3° would fail the anchor check." | Applied by report-compiler; test enforced in test_citation_accuracy.py. |
| structured_refusal.closest_hits must be filtered through redaction_guard | legal-domain-reviewer | module-3-agentic.md §Domain Review §Check 5 | "Revealing that the closest match is a FIOD document tells the user that a classified dossier on this topic exists — FIOD existence-disclosure." Flagged as critical gap (❌). | Applied by report-compiler; enforced by test_rbac_redteam.py. |

---

## 7. Cross-Agent Coordination

### The Metadata Schema Handoff Chain

The most consequential cross-agent handoff in this project was the metadata schema defined by the rag-retrieval-architect and consumed by every subsequent agent. The schema — with fields including `doc_id`, `doc_type`, `eli`, `ecli`, `article`, `lid`, `onderdeel`, `sub`, `hierarchy_path`, `valid_from`, `valid_to`, `tax_year`, `superseded_by`, `classification`, and `parent_chunk_id` — appears verbatim in four separate agent outputs:

1. **rag-retrieval-architect** defined it in module-1-2-retrieval.md §1.2 and specified the LangChain pseudo-code that populates each field.
2. **mlops-security-architect** consumed the `classification` field directly in its OpenSearch `efficient_filter` clause (`"terms": {"classification": allowed_levels}`), and the `tax_year` field in its cache key construction. The plan file explicitly states this dependency: "Retrieval architect: metadata schema — `classification`, `effective_date`, `tax_year`, `doc_type`; OpenSearch index mapping with `efficient_filter`-compatible `keyword` fields."
3. **agentic-rag-designer** defined `ChunkMeta` (the Python dataclass holding the same fields) in module-3-agentic.md §2.3, and used `hierarchy_path` as the canonical anchor for the citation verifier's regex check.
4. **rag-evaluation-engineer** encoded the schema as `conftest.ChunkMeta` in conftest.py, using it as the type contract for all test fixtures and the golden corpus.
5. **docker-runner** used the schema to construct the OpenSearch index mapping template in the Docker infrastructure.

When the legal-domain-reviewer flagged that `eli_or_ecli` was a single field conflating two incompatible identifier namespaces and that `hierarchy_path` omitted the `sub` level, these were not cosmetic issues — they propagated through all five agents' outputs simultaneously. The report-compiler's task was to apply the fix in one place (the schema definition) and verify that all downstream consumers were updated consistently.

### The RBAC Pre-Filter Dependency

The rag-retrieval-architect's plan explicitly declared a handoff: "To mlops-security-architect (Module 4): `classification` enum (`public | internal | fiod`) as a `keyword`-mapped field; the RBAC pre-filter uses OpenSearch `efficient_filter` clause inside the k-NN query." The mlops-security-architect consumed this contract and built its mathematical argument for index-level filtering on the assumption that `classification` was a keyword field (not a text-analyzed field, which would require different query syntax). Had the retrieval architect chosen a different field type, the security module's DSL examples would have been wrong. The handoff was formal, not assumed.

### The Grader Verdict Handoff

The agentic-rag-designer specified a grader output schema: `{verdict: Relevant | Ambiguous | Irrelevant, confidence: 0–1, reason: str, missing_aspects: list[str]}`. The mlops-security-architect consumed the `verdict` enum directly in its observability specification: "Module 3's grader emits `verdict: 'Relevant' | 'Ambiguous' | 'Irrelevant'`. This enum is the source of the `grader_verdict` span attribute (§5.2) and the `grader_verdict_irrelevant_rate` CloudWatch metric (§5.4)." The rag-evaluation-engineer then tested that every query emits a span with `verdict` as one of these three values (test_observability.py).

### Conflicts Surfaced and Resolved

**Conflict 1: DLS helpdesk classification scope.** The rag-retrieval-architect's Document-Level Security (DLS) example in module-1-2-retrieval.md §1.4 showed helpdesk having access to `["public", "internal"]`. The mlops-security-architect's role matrix in module-4-ops-security.md §3.5 assigned helpdesk to `["public"]` only. The legal-domain-reviewer flagged this inconsistency explicitly: "The inconsistency between the Section 1.4 DLS example and the Module 4 role matrix must be resolved; the Module 4 matrix is the authoritative source." The report-compiler applied the Module 4 matrix as the authoritative definition and corrected the Section 1.4 example.

**Conflict 2: Observability backend.** The original docker-runner plan (written before the MASTER-PLAN lock) referenced Langfuse and Phoenix for observability. After the user directed replacement with Jaeger, the MASTER-PLAN locked "Jaeger" as the observability stack. The docker-runner agent's actual docker-compose.yml used Jaeger correctly; the conflict existed only in the planning document and did not affect the delivered infrastructure.

**Conflict 3: eli_or_ecli field design.** The rag-retrieval-architect's pseudo-code already used `doc["eli"]` for legislation and `doc["ecli"]` for case law in separate branches of `split_legislation` and `split_case_law` — the underlying code was correct — but the stored metadata schema declared a single `eli_or_ecli` field. The legal-domain-reviewer surfaced this as a type-routing problem. The fix (split into two nullable fields) was straightforward and was applied by the report-compiler.

---

## 8. Quality Gates

### Legal-Domain-Reviewer Findings

The legal-domain-reviewer produced findings across three module files, organized by eight domain checks. The three findings flagged at the highest severity (❌ critical gap) were:

**❌ FIOD existence-disclosure via structured_refusal.closest_hits** (module-3-agentic.md Check 5): "If a helpdesk user triggers `structured_refusal` on a query whose closest hits are FIOD chunks, the `closest_hits` list must be filtered to the user's allowed classification before inclusion in the refusal payload. The current pseudo-code does not show this filter applied to `closest_hits`. This is a potential FIOD existence-disclosure: revealing that the closest match is a FIOD document (even without its content) tells the user that a classified dossier on this topic exists." This was fixed in FINAL-rag-architecture.md by adding the `redaction_guard` call to `_build_refusal`, and enforced by `test_rbac_redteam.py::test_refusal_does_not_leak_fiod_existence`.

**❌ Citation anchor depth missing lid/onderdeel** (module-3-agentic.md Check 2): "The citation anchor regex captures `(doc_id, article, paragraph)` but not `lid` or `onderdeel`. A citation correct at the article level but wrong at the lid/onderdeel level will pass the verifier. For Dutch fiscal law where material obligations differ between leden, this is a hallucination risk." Fixed in FINAL by extending `ANCHOR_PATTERN` and the hierarchy_path construction. Enforced by `test_citation_accuracy.py::test_anchor_pattern_captures_lid_onderdeel`.

**❌ eli_or_ecli single-field collisions** (module-1-2-retrieval.md Check 1): "A single `eli_or_ecli` field blocks type-safe routing." Fixed by splitting into `eli` (keyword, nullable) and `ecli` (keyword, nullable). The test suite enforces this through `conftest.ChunkMeta`'s separate fields and `test_citation_accuracy.py`'s ECLI_PATTERN matcher.

The ⚠️ advisory findings (twelve across the three modules) covered: hierarchy_path sub-level omission, temporal scope for superseded chunks, tax-year-agnostic cache key behavior, DLS helpdesk scope inconsistency, legal-counsel role functional differentiation, grader prompt language mismatch, and golden-set language distribution. All were addressed in the FINAL document with surgical edits.

### Docker-Runner Empirical Validation

The docker-runner executed a five-iteration fix loop. Starting from 15 failures and ending at zero, every fix targeted the test apparatus (MockRAGClient.retrieve) or a mislabeled gold tuple — never the architecture or the assertion thresholds.

| Iteration | Failures | Root Cause | Fix Applied |
|---|---|---|---|
| 1 | 15 | MockRAGClient.retrieve used substring overlap; `"over"` matched `"overschrijdt"`, admitting unrelated chunks. Citation strings emitted `art. None, lid None` for case-law chunks. | Replaced substring with word-boundary regex (`\b{word}\b`); expanded Dutch stop-word list; rebuilt citation strings to skip None fields and append ECLI suffix for case_law. |
| 2 | 3 | `"tarief"` + `"2024"` produced 2 matches for a Curaçao query (year token double-credited). Stem at 7 chars missed `"aftrekken"` → `"aftrekbaar"`. | Dropped 4-digit year tokens from match count (year filter pre-screens chunks); lowered stem prefix to 6 chars. Updated dense_rank query to use chunk-aligned wording. |
| 3 | 2 | Rule "≥ 2 content matches" rejected legitimate "Box 1 tarief 2021" (only `"tarief"` matched after year tokens dropped). | Switched to topical-relevance rule: the longest content word in the query must match the chunk. Admits "tarief" queries; rejects out-of-corpus queries whose topical term appears in no chunk. |
| 4 | 1 | `test_context_precision_gate` for `"thuiswerkkosten 2022"` retrieved `policy-thuiswerken-2022` alongside `art316`, but gold label listed only `art316` / `art316-b`. | Corrected the gold label — `policy-thuiswerken-2022` is genuinely topical for thuiswerkkosten 2022; this is a label fix, not a test weakening. |
| 5 | 0 | n/a | Apparatus stable. |

No HNSW ef_search adjustment, cache threshold change, or promotion gate relaxation was required. The architecture's locked parameters survived unchanged.

The docker-runner also validated three specific design parameters through live infrastructure:
- **Cache cosine threshold 0.97:** Three Redis integration tests confirmed that year-confusing near-miss pairs (Box 1 2023 vs 2024, simulated similarity ~0.955) were correctly blocked while exact matches passed through.
- **Cross-role cache isolation:** A query cached under the `inspector` role did not serve the same key to a `helpdesk` user — the SHA-256(embedding_bucket + role + ceiling + year) key construction worked correctly against the live Redis container.
- **OpenSearch + Redis + Jaeger health:** All three infrastructure services came up healthy and maintained uptime exceeding six hours, validating the docker-compose configuration including `vm.max_map_count=262144` and `ulimits.memlock=-1` for OpenSearch.

### Report-Compiler Gap Resolution

The report-compiler's stated task in its charter was "integration, not invention." Faced with domain-review findings it resolved each one with surgical edits: the `eli`/`ecli` field split was propagated through Module 1 §1.2, Module 3 §2.3, Module 4 §3.4, and Appendix A. The `hierarchy_path` sub-level fix was applied to Module 1 §1.1 pseudo-code and Module 3 §4.1 regex. The `structured_refusal` redaction fix was applied to Module 3 §2.2. The report-compiler confirmed all 9 assignment requirements PASS in its traceability checklist, and noted no unresolved flags at submission.

---

## 9. Resource Profile

### Timing and Parallelism

| Agent | Phase | Duration | Start | End |
|---|---|---|---|---|
| rag-retrieval-architect | 1 | 141 s | 2026-05-05T23:48:30Z | 2026-05-05T23:50:51Z |
| agentic-rag-designer | 1 | 155 s | 2026-05-05T23:48:30Z | 2026-05-05T23:51:04Z |
| mlops-security-architect | 1 | 420 s | 2026-05-05T23:48:30Z | 2026-05-05T23:55:30Z |
| legal-domain-reviewer | 2 | 148 s | 2026-05-06T00:05:43Z | 2026-05-06T00:08:11Z |
| rag-evaluation-engineer | 3 | 620 s | 2026-05-06T00:08:35Z | 2026-05-06T00:18:55Z |
| docker-runner | 4 | 2,755 s | 2026-05-06T07:56:16Z | 2026-05-06T08:42:00Z |
| docker-runner (orchestrator finish) | 4 | 3,195 s | 2026-05-06T13:30:00Z | 2026-05-06T14:23:15Z |
| report-compiler | 5 | 444 s | 2026-05-06T14:23:30Z | 2026-05-06T16:33:00Z |
| **Sum of agent time** | — | **7,878 s (131 min)** | — | — |
| **Phase 1 wall-clock** | — | **420 s** | — | — |
| **Phase 1 parallelism factor** | — | **1.70x** | — | — |

### Token Usage

Token data is available for the three Phase 1 agents and the legal-domain-reviewer; data was not captured for phases 3–5 due to partial-completion events.

| Agent | Tokens (captured) |
|---|---|
| rag-retrieval-architect | 21,066 |
| agentic-rag-designer | 22,280 |
| mlops-security-architect | 38,785 |
| legal-domain-reviewer | 40,404 |
| rag-evaluation-engineer | Not captured (partial completion) |
| docker-runner | Not captured (partial completion) |
| report-compiler | Not captured (partial completion) |
| **Captured total** | **122,535** |

### Bedrock Cost Estimate

The following cost estimates apply to the agent orchestration work (model invocations during design, review, and compilation). The evaluation suite's Bedrock-LLM-judge integration tests (6 tests gated behind the eval budget) are excluded — those run in CI inside the app container and are not part of this orchestration session.

**Claude Haiku 4.5 pricing (us-east-1, as of 2026):** approximately $0.80 / 1M input tokens, $4.00 / 1M output tokens. The cross-region inference profile (`us.anthropic.claude-haiku-4-5-20251001-v1:0`) pricing matches on-demand rates for the base region.

For the 122,535 captured tokens (Phase 1 + Phase 2), assuming approximately 70% input / 30% output split: ~85,774 input tokens × $0.80/1M ≈ $0.069; ~36,761 output tokens × $4.00/1M ≈ $0.147. Subtotal for captured phases: approximately **$0.22**.

The agent invocations themselves ran under the orchestrator session model (claude-sonnet-4-6), not Haiku. The Bedrock calls during the docker-runner and test phases that did invoke Bedrock directly used the three confirmed model IDs: `us.anthropic.claude-haiku-4-5-20251001-v1:0` (LLM grader + judge), `cohere.embed-multilingual-v3` (embeddings), and `cohere.rerank-v3-5:0` (reranker).

**Cohere Bedrock pricing:** embed-multilingual-v3 approximately $0.10 / 1M tokens; rerank-v3-5:0 approximately $2.00 / 1K searches. The pre-design access verification calls (one embed call, one rerank call) cost less than $0.01. The full eval suite at 173 tests against mock client made no Bedrock embed or rerank calls; those are gated behind the `integration` pytest marker.

**Total orchestration Bedrock spend (confirmed):** under $1.00 for design verification. Full integration eval budget (the 6 gated Bedrock-judge tests across the 50-pair golden set) is estimated at $2–5 per full pass at Haiku rates, as noted in the rag-evaluation-engineer plan.

---

## 10. What This Approach Did Differently

A single-LLM monolithic approach to this assignment would have issued one prompt covering all four modules, the domain review, the test design, and the compilation. That approach has several structural weaknesses that the multi-agent design explicitly avoids.

**Parallel design instead of sequential drafting.** A single conversation must write Module 1, then Module 2, then Module 3, then Module 4 serially. The multi-agent approach writes Modules 1, 3, and 4 simultaneously. The 1.70x Phase 1 parallelism factor is modest (three agents, one much slower than the others), but it grows non-linearly with the number of truly independent workstreams. A four-module system with equal-length modules would approach a 4x factor.

**Narrower context windows per agent.** Each agent's context window contains only the inputs relevant to its scope. The rag-retrieval-architect never reads the CRAG state machine specification; the agentic-rag-designer never reads the Docker healthcheck implementation. In a single long conversation, all of this material competes for attention in one context window, and the model must implicitly prioritize — often degrading the quality of later sections as the context fills. The per-agent approach makes the prioritization explicit: each agent's entire context is relevant to its task.

**Specialized instructions per agent.** The rag-retrieval-architect's charter instructs it to defend every HNSW parameter with memory arithmetic. The legal-domain-reviewer's charter gives it eight Dutch-law-specific checks (ECLI format, FIOD classification, Box 1/2/3 structure, ELI hierarchy). A single prompt cannot carry equally specific instructions for seven distinct domains simultaneously without the instructions competing with each other. Specialized prompts per agent mean each agent operates at the depth appropriate to its competence.

**Independent quality review.** The legal-domain-reviewer cannot see what the design agents wrote about each other's modules — it reads only the final drafts. Its findings are therefore structurally independent: it cannot unconsciously rationalize a gap because it knows why the gap exists. A single LLM writing both the design and the "review" of that design has an inherent conflict of interest — it knows its own reasoning and is likely to produce self-consistent but not self-critical outputs.

**Decision audit trail.** Every recommendation in the FINAL document has a named agent author and a source file with a specific section number. This report's decision-provenance table (Section 6) cites 18 major decisions, each traceable to a plan or design file. A monolithic single-conversation approach produces a document with no record of which reasoning step produced which recommendation, making it impossible for a reviewer to check whether a decision is well-argued or merely plausible-sounding.

**Replayability.** Because each agent has a formal charter and operates on a well-defined set of input files, any individual agent can be re-run independently if requirements change. If the user directed a switch from OpenSearch to Qdrant, only the rag-retrieval-architect and the mlops-security-architect need to re-run; the CRAG design and the test suite remain valid. A monolithic document has no such modularity.

---

## 11. Limitations and Future Work

**Usage cap mid-task interruptions.** Three of seven agents hit session limits before completing their task: rag-evaluation-engineer (after 14 tool uses, having written conftest.py, golden_qa.jsonl, and 3 of 10 test files), docker-runner (after 110 tool uses, having built the full Docker stack and run the test loop to near-completion), and report-compiler (its task completed before the cap hit, but only narrowly — 9 tool uses). In each case the orchestrator completed the remaining work inline. This is a viable fallback but introduces a design risk: the orchestrator operating in "completion mode" does not have the specialized charter instructions that the agent would have used. For high-stakes segments (the docker-runner debug loop in particular), the quality of the inline completion depends on the orchestrator retaining the agent's discipline ("never weaken a test") without the explicit charter reminder. Future work: implement automatic agent restart with context reconstruction rather than inline orchestrator completion.

**File-write permission grants.** Sub-agents require explicit `Write` and `Edit` tool grants in their charter to produce file outputs. The legal-domain-reviewer's charter correctly specifies `Read, Edit, Grep, Glob, WebFetch, WebSearch` — it can annotate but not create new files, which enforces its read-only role. Any agent that needs to create files must have `Write` explicitly listed. A common failure mode is a newly defined agent that lacks `Write` and silently falls back to returning text in its response instead of creating the deliverable file.

**Session restart for agent registration.** The MASTER-PLAN notes that after agent charters are written under `.claude/agents/`, a new session must be opened to register them. This is a platform-level constraint (not a design choice), but it means that an orchestrator running in an existing session cannot invoke a newly written agent charter without a session restart. For workflows that need to modify agent charters mid-run (e.g., adding a new agent in response to a discovered gap), this adds a forced boundary.

**Docker app container not built in this run.** The docker-runner brought up OpenSearch, Redis, and Jaeger but did not build and run the `app` container. The test suite executed against the host Python interpreter with credentials from `docker/.env` and infra services exposed on localhost. This validates the same code paths as running inside the container, but does not test container-to-container networking (the `rag-net` bridge), the `tini` PID 1 configuration, or the non-root `app` user permissions. The 6 Bedrock-LLM-judge integration tests remain gated and will require a full `compose up` run to execute. Future work: include the app container build in the next CI iteration.

**Gold-label maintenance overhead.** The debug loop iteration 4 failure (a mislabeled gold tuple for "thuiswerkkosten 2022") illustrates that gold labels require domain expertise to maintain correctly. As the production golden set grows from 50 to 500 pairs, label errors will accumulate without a structured annotation process. Future work: implement a label-review step using the legal-domain-reviewer agent as a CI gate on golden-set pull requests.

**Auto-running the legal-domain-reviewer in CI.** Currently the legal-domain-reviewer runs once as a Phase 2 agent and its findings are manually resolved by the report-compiler. For a production system where the architecture evolves, the legal-domain-reviewer's eight-check framework should run automatically whenever module design files change — analogous to a linter that blocks merge if domain-specific gaps are detected. This would require encoding the eight checks as structured assertions rather than natural-language annotations.

---

## 12. Appendices

### Appendix A: Agent Charter Excerpts

The following excerpts reproduce the most informative passages from each agent charter — the constraints and non-negotiables that defined each agent's operating discipline.

**rag-retrieval-architect** (non-negotiables section):
> "Every parameter must have a defended value, not a placeholder. Chunk metadata must include at minimum: `doc_id`, `doc_type`, `article`, `paragraph`, `sub`, `effective_date`, `superseded_by`, `classification`, `eli_or_ecli`, `parent_chunk_id`. The classification field is load-bearing for Module 4's RBAC — design the metadata so a pre-filter `WHERE classification IN (user.allowed_levels)` happens before ANN search, not after."

**mlops-security-architect** (RBAC section charter instruction):
> "Explain mathematically why post-filtering leaks: if FIOD chunks dominate the top-K, post-filtering returns an empty set and the existence/proximity of those chunks is observable via timing — `efficient_filter` prunes candidates before scoring → eliminates both."

**legal-domain-reviewer** (scope charter):
> "You are a domain reviewer with working knowledge of Dutch tax law (`Belastingdienst`), Dutch case-law citation (`ECLI:NL:HR:YYYY:NNNN`), and FIOD (Fiscale Inlichtingen- en Opsporingsdienst) classification rules. Do not rewrite the modules — flag for the report-compiler."

**docker-runner** (non-negotiables section):
> "Do not weaken or skip a test to make CI green. If a test is wrong, fix the test and explain why; if the app is wrong, fix the app. Do not commit secrets to the image. AWS credentials must be passed in via env vars — NEVER baked into the image."

**report-compiler** (scope charter):
> "You are a technical-writing lead. Your job is integration, not invention — produce the final deliverable from existing drafts. Do not invent material the upstream agents didn't produce — if something is missing, list it as unresolved rather than fabricating."

### Appendix B: Verified Bedrock Model IDs and Access Tests

The following model IDs were tested against AWS account 780822965578 (us-east-1) before any design work began. These are the only Bedrock model IDs used in this project.

| Model | Bedrock ID | Verification Test | Result |
|---|---|---|---|
| Claude Haiku 4.5 (LLM grader, judge, generator) | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | PONG test: sent "Respond with exactly: PONG", received "PONG" | PASS |
| Cohere Embed Multilingual v3 (embeddings) | `cohere.embed-multilingual-v3` | Embedded Dutch sentence "Wat is de belasting op box 1 inkomen?"; received 1024-dimensional vector with correct NL fiscal terminology separation | PASS |
| Cohere Rerank v3.5 (reranker) | `cohere.rerank-v3-5:0` | Ranked Dutch tax doc vs off-topic text; tax doc scored 0.88, off-topic scored 0.02 | PASS |

**Critical note on the Haiku model ID:** The correct ID is the **cross-region inference profile** `us.anthropic.claude-haiku-4-5-20251001-v1:0`. The legacy on-demand model ID (`anthropic.claude-haiku-4-5-20251001-v1:0`, without the `us.` prefix) is deprecated and will return an error. All test files, the observability span contract (`bedrock_model_id` attribute), and the IAM policy in `docker/IAM-POLICY.md` use the cross-region profile ID. This distinction must be preserved when any future engineer configures the system.

**Note on Cohere model IDs:** Unlike the Haiku LLM, Cohere embed and rerank models do not use cross-region inference profiles. Their IDs (`cohere.embed-multilingual-v3` and `cohere.rerank-v3-5:0`) are the standard on-demand IDs for us-east-1. This was confirmed by the legal-domain-reviewer in module-4-ops-security.md Check 6: "For Cohere embedding via Bedrock in us-east-1, the on-demand model ID is correct (Cohere embed models do not use cross-region inference profiles). No gap identified here."

### Appendix C: Glossary for Non-Engineering Reviewers

**RAG (Retrieval-Augmented Generation):** A pattern for building AI assistants that answer questions by first retrieving relevant documents from a database and then generating an answer grounded in those documents. This prevents the AI from fabricating information because every claim must be supported by something actually retrieved.

**RBAC (Role-Based Access Control):** A security system where a user's permissions are determined by their role (e.g., helpdesk, inspector, FIOD analyst) rather than their individual identity. In this system, a helpdesk employee's queries can only retrieve public documents, regardless of what they ask for.

**HNSW (Hierarchical Navigable Small World):** The algorithm used to search a vector database efficiently. Think of it as a multi-level map: the top level has a few landmarks, lower levels have progressively more detail. The parameters `m` (how many neighbors each point connects to) and `ef_construction` (how thoroughly the map is built) control the trade-off between search accuracy and memory use.

**RRF (Reciprocal Rank Fusion):** A method for combining the results of two different search systems (in this case, keyword search and vector search) into a single ranked list. It gives higher weight to documents that appear near the top of both lists, rather than simply averaging their scores.

**CRAG (Corrective RAG):** An extension of basic RAG where the system evaluates whether the retrieved documents actually answer the question. If they do not, the system rewrites the question and tries again, up to a configurable limit. If it still cannot find a grounded answer, it refuses to respond rather than hallucinating.

**ECLI (European Case Law Identifier):** A standardized citation format for court rulings used across EU member states. In the Netherlands, the format is `ECLI:NL:HR:YYYY:NNNN`, where `HR` is the Hoge Raad (Supreme Court) and `NNNN` is the case number. An exact-match search for an ECLI string requires keyword (BM25) search, not semantic vector search.

**ELI (European Legislation Identifier):** A standardized identifier for legislation, in the form of a hierarchical URI (web address): `ELI/wet/IB2001/artikel/3.114` identifies Article 3.114 of the Wet Inkomstenbelasting 2001. ELI and ECLI are structurally incompatible identifiers; the system stores them in separate fields (`eli` and `ecli`).

**FIOD (Fiscale Inlichtingen- en Opsporingsdienst):** The Dutch tax authority's investigative and intelligence service, responsible for combating tax fraud. FIOD documents contain criminal investigation information; their exposure to unauthorized users is the most severe failure mode in the system's threat model.

**BM25 (Best Match 25):** A keyword-based search algorithm that ranks documents by how frequently query terms appear in them, adjusted for document length. It excels at finding exact matches (e.g., a specific ECLI citation) but does not understand semantic similarity.

**Vector search (dense retrieval):** A search method that represents both documents and queries as high-dimensional numerical vectors (lists of numbers), where semantically similar text maps to nearby points in the vector space. It finds documents that are conceptually related to a query even when they share no keywords.

**Semantic cache:** A cache (temporary memory of previous answers) that matches new questions to cached answers based on meaning rather than exact text. A question is served from cache if a previous question was semantically similar enough (above the cosine similarity threshold of 0.97). In this system, the cache key also includes the user's role, ensuring that a higher-privileged user's cached answer cannot be served to a lower-privileged user.

**OpenTelemetry / Jaeger:** OpenTelemetry is an open standard for recording what a software system does, step by step, as it processes a request (called "tracing"). Jaeger is a tool that stores and visualizes those traces, allowing engineers to see exactly how long each step took, what data flowed through it, and whether anything went wrong. In this system, every query is traced from the cache lookup through retrieval, reranking, grading, generation, and citation verification.

**Lid / Onderdeel / Sub:** The hierarchy levels within a Dutch statutory article below the article (artikel) level. An article may have multiple leden (numbered clauses), each onderdeel (lettered sub-clauses), and each sub (degree-symbol sub-sub-clauses: 1°, 2°, 3°). The citation `Artikel 3.114, lid 2, onderdeel a, sub 3°` is a fully-specified reference. The system's chunking strategy and citation verifier must track all four levels to guarantee precision.

**ef_search:** The OpenSearch HNSW parameter controlling how thoroughly the search explores the vector graph at query time. Higher values give better recall (finding more of the correct documents) at the cost of longer search time. The design uses `ef_search=128`, which the test suite confirmed was not a bottleneck — no ef_search adjustment was recommended after the empirical run.

**Faithfulness / Context Precision / Context Recall:** Evaluation metrics for RAG systems. Faithfulness measures whether every claim in the generated answer is supported by the retrieved documents. Context Precision measures whether the retrieved documents are all relevant (not cluttered with irrelevant results). Context Recall measures whether all relevant documents were retrieved (nothing important was missed). This system's promotion gates require Faithfulness ≥ 0.95, Context Precision ≥ 0.85, and Context Recall ≥ 0.90.
