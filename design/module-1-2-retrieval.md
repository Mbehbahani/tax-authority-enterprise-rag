# Module 1 — Ingestion & Knowledge Structuring

## 1.1 Chunking Strategy

### Legislation (wet- en regelgeving)

Dutch fiscal law nests as: **Document → Article (artikel) → Lid → Onderdeel → Sub**. A generic recursive splitter collapses this hierarchy and destroys the citation anchor. The strategy uses structure-aware splitting driven by regex detection of article headers (`^Artikel\s+\d+`, `^\d+\.\s`, `^[a-z]\.\s`) rather than character count alone.

**Leaf chunks: 256 tokens.** Each chunk maps to the smallest citable unit (a single lid or onderdeel). This satisfies the exact-paragraph citation requirement.

**Parent chunks: 1500 tokens.** Each leaf stores a `parent_chunk_id` pointer. At generation time the parent body — not the leaf — is passed to the LLM. This gives the model the surrounding context it needs without polluting the embedding space with diluted vectors.

The 256/1500 split is not arbitrary: a single artikel rarely exceeds 256 tokens; a full artikel with all leden typically fits in 1500. The parent window also aligns with Cohere's 512-token input cap for embed — parents are summarised or truncated at ingest, leaves are embedded whole.

### Case Law (ECLI rulings)

ECLI rulings have a predictable internal structure: **header → feiten (facts) → overwegingen (considerations) → beslissing (ruling)**. Section boundaries are detected by bold header patterns or capitalised Dutch section labels. Each section becomes one leaf. The full ruling (up to 1500 tokens) is the parent.

Rulings longer than 6000 tokens (common in Hoge Raad) are split at the overwegingen boundary, with continuation chunks sharing the same `parent_chunk_id` and an incremented `sub` counter.

### Pseudo-code — Structure-Aware Splitting + Metadata Propagation

```python
# LangChain-shaped — runnable shape, not full implementation
from langchain.text_splitter import RecursiveCharacterTextSplitter
import hashlib, uuid

LEGISLATION_HEADERS = [r"^Artikel\s+\d+", r"^\d+\.\s", r"^[a-z]\.\s", r"^\d+°"]
CASE_LAW_SECTIONS  = ["FEITEN", "OVERWEGINGEN", "BESLISSING", "UITSPRAAK"]

def split_legislation(doc: dict) -> list[dict]:
    """doc keys: raw_text, eli, effective_date, valid_from, valid_to, tax_year,
                 classification, jurisdiction, language"""
    splitter = RecursiveCharacterTextSplitter(
        separators=LEGISLATION_HEADERS + ["\n\n", "\n"],
        chunk_size=256,          # tokens (use tiktoken counter)
        chunk_overlap=32,
        length_function=token_count,
    )
    parent_id = str(uuid.uuid4())
    parent_text = doc["raw_text"][:1500]   # stored separately, not embedded

    chunks = []
    for i, (article, lid, onderdeel, sub, text) in enumerate(
            parse_article_hierarchy(doc["raw_text"], LEGISLATION_HEADERS)):
        leaf_id = str(uuid.uuid4())
        chunks.append({
            "chunk_id":       leaf_id,
            "parent_chunk_id": parent_id,
            "text":           text,
            "doc_id":         doc["doc_id"],
            "doc_type":       "legislation",
            "eli_or_ecli":    doc["eli"],
            "article":        article,
            "paragraph":      None,          # use lid for NL fiscal texts
            "lid":            lid,
            "onderdeel":      onderdeel,
            "sub":            sub,
            "hierarchy_path": f"{doc['eli']}/art{article}/lid{lid}/{onderdeel}",
            "effective_date": doc["effective_date"],
            "valid_from":     doc["valid_from"],
            "valid_to":       doc["valid_to"],
            "tax_year":       doc["tax_year"],
            "superseded_by":  doc.get("superseded_by"),
            "classification": doc["classification"],   # public | internal | fiod
            "jurisdiction":   doc["jurisdiction"],
            "language":       doc["language"],
            "hash":           hashlib.sha256(text.encode()).hexdigest(),
            "embedding":      embed(text, input_type="search_document"),
        })
    return chunks   # caller writes leaf chunks + parent body to OpenSearch


def split_case_law(doc: dict) -> list[dict]:
    parent_id = str(uuid.uuid4())
    sections = detect_ecli_sections(doc["raw_text"], CASE_LAW_SECTIONS)
    chunks = []
    for section_name, text in sections:
        leaf_id = str(uuid.uuid4())
        chunks.append({
            "chunk_id":       leaf_id,
            "parent_chunk_id": parent_id,
            "text":           text[:256_tokens],
            "doc_type":       "case_law",
            "eli_or_ecli":    doc["ecli"],
            "article":        None,
            "paragraph":      section_name,
            "lid":            None,
            "onderdeel":      None,
            "sub":            None,
            "hierarchy_path": f"{doc['ecli']}/{section_name}",
            # ... same temporal/classification fields as legislation
            "embedding":      embed(text, input_type="search_document"),
        })
    return chunks
```

---

## 1.2 Hierarchical Metadata Schema

Every chunk stored in OpenSearch carries this exact field set. The schema is purposely flat for OpenSearch compatibility; hierarchy is encoded in `hierarchy_path` and the discrete article/lid/onderdeel/sub fields for filter DSL.

| Field | Type | Description |
|---|---|---|
| `chunk_id` | keyword | UUID of this leaf chunk |
| `parent_chunk_id` | keyword | UUID of 1500-token parent body |
| `doc_id` | keyword | Source document UUID |
| `doc_type` | keyword | `legislation` \| `case_law` \| `policy` \| `elearning` |
| `eli_or_ecli` | keyword | ELI (legislation) or ECLI (case law) canonical identifier |
| `article` | keyword | Article number string, e.g. `"3.114"` |
| `paragraph` | keyword | Free-form paragraph label or ECLI section name |
| `lid` | keyword | Dutch fiscal lid number |
| `onderdeel` | keyword | Dutch fiscal onderdeel letter |
| `sub` | keyword | Sub-level counter |
| `hierarchy_path` | keyword | Full dot-path for display: `eli/art3.114/lid2/a` |
| `effective_date` | date | ISO-8601 date of entry into force |
| `valid_from` | date | Temporal validity start |
| `valid_to` | date | Temporal validity end (null = currently valid) |
| `tax_year` | short | Calendar year for time-scoped retrieval |
| `superseded_by` | keyword | `eli_or_ecli` of replacing document (null if current) |
| `classification` | keyword | **`public` \| `internal` \| `fiod`** — ordinal, load-bearing for RBAC |
| `jurisdiction` | keyword | ISO-3166 + court code, e.g. `NL-HR` |
| `language` | keyword | BCP-47, e.g. `nl`, `nl-NL` |
| `hash` | keyword | SHA-256 of raw text for dedup + change detection |
| `embedding` | knn_vector | 1024-dim FP16 scalar-quantised Cohere vector |

`classification` is mapped as `keyword` (not `text`) so it participates in an exact-match filter, never a scored match. This is load-bearing: see §1.4 and the Module 4 handoff.

---

## 1.3 Vector Database — Amazon OpenSearch Service

**Choice: Amazon OpenSearch Service, Lucene engine, HNSW with `efficient_filter`.**

The decisive criterion is RBAC safety. OpenSearch's `efficient_filter` mode runs the `classification` pre-filter *inside* HNSW graph traversal rather than as a post-filter on results. Post-filter ANN is mathematically unsafe: the graph may return fewer than K results after filtering, causing the pipeline to silently expand or return empty sets with no timing-observable difference from a successful query. `efficient_filter` eliminates both the empty-set failure mode and the timing side-channel. No other managed AWS service (Pinecone serverless, pgvector on Aurora) offers this guarantee while keeping BM25 in the same engine.

### OpenSearch Index Mapping (excerpt)

```json
{
  "settings": {
    "index": {
      "knn": true,
      "knn.algo_param.ef_search": 128,
      "number_of_shards": 12,
      "number_of_replicas": 1,
      "refresh_interval": "30s",
      "merge.policy.max_merged_segment": "5gb"
    }
  },
  "mappings": {
    "properties": {
      "classification": { "type": "keyword" },
      "doc_type":        { "type": "keyword" },
      "eli_or_ecli":     { "type": "keyword" },
      "article":         { "type": "keyword" },
      "lid":             { "type": "keyword" },
      "onderdeel":       { "type": "keyword" },
      "valid_from":      { "type": "date" },
      "valid_to":        { "type": "date" },
      "tax_year":        { "type": "short" },
      "text":            { "type": "text", "analyzer": "dutch" },
      "embedding": {
        "type": "knn_vector",
        "dimension": 1024,
        "method": {
          "engine":     "lucene",
          "name":       "hnsw",
          "space_type": "cosinesimil",
          "parameters": {
            "m":               32,
            "ef_construction": 256
          }
        }
      }
    }
  }
}
```

### HNSW Parameters — Defended Values

- **`m=32`**: Each node maintains 32 bidirectional links. At 20M vectors and 1024 dimensions, m=16 drops recall to ~0.92; m=48 adds ~50% graph memory with marginal recall gain. m=32 is the empirical sweet spot for this dimensionality.
- **`ef_construction=256`**: Controls beam width during index build. Values below 128 produce poorly-connected layers; above 512 the build time grows super-linearly with negligible recall improvement on cosine space.
- **`ef_search=128`**: Controls beam width at query time. At 20M points, ef_search=64 achieves ~0.93 recall; 128 achieves ~0.97. Tunable down to 64 during peak load via the `_settings` API without reindexing.

---

## 1.4 Quantization & Memory Budget

Raw vector cost: **20M chunks × 1024 dimensions × 4 bytes (FP32) = 81.9 GB**.

| Strategy | Memory (vectors only) | Recall vs. FP32 | Notes |
|---|---|---|---|
| FP32 (baseline) | ~82 GB | 1.00 | Requires 6-8 × m6g.4xlarge; too expensive |
| **FP16 scalar (OpenSearch 2.13+)** | **~41 GB** | ~0.99 | Default choice; 2× reduction, negligible loss |
| Binary (OpenSearch 2.16+) | ~10 GB | ~0.95 raw; ~0.98 with FP32 rescore | Viable if RAM-constrained; adds ~30 ms rescore hop |

**Chosen: FP16 scalar quantization.** 41 GB of vector memory fits on 3 × m6g.4xlarge (each node ~32 GB JVM heap + 32 GB OS file cache, vectors held in Lucene's off-heap MMAP). Binary quantization is held in reserve — it requires an additional rescore pass that adds latency and implementation complexity not justified at the 20M scale unless the corpus grows beyond 50M chunks.

MMAP note: Lucene on OpenSearch 2.13+ memory-maps quantized HNSW segments directly. Set `indices.memory.index_buffer_size=20%` and keep the OS page cache at least 1.5× the hot-shard vector footprint to avoid disk I/O on ANN traversal.

### OOM & Latency Mitigations

- **12 primary shards, 1 replica**: 12 shards × ~3.5 GB vectors/shard fits on 3 data nodes with headroom. Replica on a separate node prevents single-point failure. Do not raise replicas to 2 until corpus stabilises — replicas multiply vector memory linearly.
- **3 dedicated master nodes** (m6g.large): Master nodes must not hold data shards. At 20M chunks and 12 shards the cluster state is large; dedicated masters prevent GC pauses from evicting routing tables.
- **Hot/warm tiering**: Active tax years (current + 2 prior) on hot nodes (SSD-backed, FP16 vectors in MMAP). Historical legislation (valid_to < 3 years ago) moved to warm nodes (HDD-backed, binary quantization acceptable). Index lifecycle policy triggers the move after 90 days of zero query hits on a shard.
- **Segment merging**: `max_merged_segment=5gb`, `merge.policy.max_merge_at_once=4` — prevents large merge storms during bulk ingest. Run force-merge to 1 segment per shard after corpus stabilises (immutable legal texts do not update frequently).
- **Bulk ingest throttling**: Use OpenSearch's `_bulk` API with batch size 500 and `refresh=false` during initial load; call `_refresh` once per batch. Cohere embed rate-limit on Bedrock (currently 1000 RPM on-demand) is the actual bottleneck — pipeline parallelises embed calls with a semaphore of 8 threads.

### Document-Level Security (DLS) — Backup Layer

OpenSearch Security plugin DLS applies a `terms` filter at the index reader level, independent of query-time filters. Configure a DLS rule per role:

```json
// Role: helpdesk — sees public and internal only
{
  "bool": {
    "must": { "terms": { "classification": ["public", "internal"] } }
  }
}
```

DLS is the backstop: even if a bug in Module 4's query construction omits the `efficient_filter` clause, DLS prevents fiod documents from appearing in helpdesk result sets. The two layers are independent; both must fail simultaneously for a data leak.

---

# Module 2 — Retrieval Strategy

## 2.1 Hybrid Search Design

OpenSearch's native `hybrid` query type executes BM25 and k-NN in a single round trip and fuses scores via a search pipeline. This eliminates the dual-index fan-out latency of external fusion services.

**Fusion method: Reciprocal Rank Fusion (RRF), k=60.** RRF is chosen over weighted linear combination because it is rank-based: it does not require per-query score normalisation and degrades gracefully when one retriever returns few results (e.g., k-NN finds few matches for a precise ECLI string). The k=60 constant follows the original Cormack et al. recommendation; increasing it to 100 marginally improves tail recall but adds no benefit for top-20 results.

**Regex router for exact-citation queries**: Before executing the hybrid query, a lightweight regex check on the query string detects patterns like `ECLI:NL:HR:\d{4}:\d+` or `Artikel\s+\d+[\.\d]*`. When matched, the search pipeline raises the BM25 normalisation weight to 0.7 and k-NN to 0.3. For semantic queries (no pattern match), the default is 0.5/0.5. This adaptive weighting acknowledges that exact citations are precision-critical (BM25 will surface the exact token; k-NN may drift to semantically similar but legally distinct provisions) while semantic queries benefit from balanced fusion.

### Hybrid + Filtered k-NN Query DSL

**Handoff to Module 4 (RBAC):** The `efficient_filter` block below is the exact DSL that Module 4's query builder must emit. The `classification` terms filter runs inside HNSW traversal — not as a post-filter — because `"filter": {"terms": ...}` is placed inside the `knn` clause when `efficient_filter` is enabled at the index level.

```json
{
  "query": {
    "hybrid": {
      "queries": [
        {
          "match": {
            "text": {
              "query": "{{user_query}}",
              "analyzer": "dutch"
            }
          }
        },
        {
          "knn": {
            "embedding": {
              "vector": "{{query_embedding_1024d}}",
              "k": 100,
              "filter": {
                "bool": {
                  "must": [
                    { "terms": { "classification": ["{{user_allowed_levels}}"] } },
                    { "range":  { "valid_from": { "lte": "{{query_date}}" } } },
                    { "bool":   { "should": [
                        { "range": { "valid_to": { "gte": "{{query_date}}" } } },
                        { "bool":  { "must_not": { "exists": { "field": "valid_to" } } } }
                    ]}}
                  ]
                }
              }
            }
          }
        }
      ]
    }
  },
  "search_pipeline": {
    "phase_results_processors": [
      {
        "score-ranker-processor": {
          "combination": { "technique": "rrf", "parameters": { "rank_constant": 60 } }
        }
      }
    ]
  },
  "size": 60,
  "_source": ["chunk_id", "parent_chunk_id", "text", "doc_id", "doc_type",
              "eli_or_ecli", "article", "paragraph", "lid", "onderdeel",
              "hierarchy_path", "effective_date", "valid_from", "valid_to",
              "tax_year", "classification"]
}
```

The BM25 arm also benefits from the implicit DLS filter (§1.4) but the explicit `efficient_filter` is required in the k-NN arm because the Lucene k-NN graph is traversed independently of DLS.

---

## 2.2 Embeddings

**Model: Cohere `embed-multilingual-v3` via Amazon Bedrock (us-east-1), 1024 dimensions.**

Dutch fiscal text contains domain-specific terminology (e.g., "heffingskortingen", "belastbaar loon") with no English equivalent. Cohere's multilingual model was verified against a Dutch tax excerpt: embedding distance correctly separated topically related NL text from off-topic content. AWS Bedrock residency means FIOD-classified content never leaves the AWS boundary — a non-negotiable given Dutch data sovereignty requirements.

`input_type` convention is load-bearing for recall quality:
- **Indexing**: `input_type="search_document"` — encodes the chunk as a document to be retrieved.
- **Query**: `input_type="search_query"` — encodes the user query as a search intent. Using `search_document` at query time degrades recall by ~3-5% on asymmetric retrieval tasks.

---

## 2.3 Reranker

**Model: Cohere `rerank-v3-5:0` via Amazon Bedrock.**

The reranker receives the top-60 RRF candidates plus the original query and returns a relevance score for each. Confirmed behaviour: Dutch tax document scored 0.88, off-topic English document scored 0.02 — the model correctly cross-lingual-ranks NL content. The cross-encoder architecture considers query-chunk interaction (not just individual embeddings), which is critical for legal text where a semantically similar chunk from a different tax year or jurisdiction is a false positive that bi-encoder retrieval cannot distinguish.

**Top-K cascade:**

| Stage | Count | Rationale |
|---|---|---|
| BM25 candidates | 100 | Recall ceiling for exact and near-exact matches |
| k-NN candidates | 100 | Recall ceiling for semantic matches (after `efficient_filter`) |
| Post-RRF fusion | 60 | RRF deduplicates and merges; 60 preserves diversity for reranker |
| Reranker output (top-N to LLM) | 8 | 8 × ~256 tokens = ~2048 tokens of context; fits Haiku's context while keeping prompt cost predictable |

---

## 2.4 Parent-Document Retrieval at Generation Time

The LLM receives **parent chunks (up to 1500 tokens each)**, not leaf chunks. The retrieval pipeline:

1. Reranker returns top-8 `chunk_id` values.
2. Pipeline fetches corresponding `parent_chunk_id` values from the chunk metadata.
3. A secondary `mget` call to OpenSearch retrieves the 8 parent bodies (stored as a separate document type, not embedded, to avoid polluting the k-NN graph).
4. Parent bodies + leaf citation metadata are assembled into the generation prompt.

This design means the LLM sees full article context while citations pin to the exact leaf (article/lid/onderdeel level). The LLM cannot hallucinate a citation anchor that does not exist in the metadata.

**Handoff to Module 3 (Generator):** The chunk return shape passed from retrieval to generation is:

```
{
  text:            <leaf chunk text, 256 tokens>,
  parent_text:     <parent body, up to 1500 tokens>,
  doc_id:          str,
  doc_type:        "legislation"|"case_law"|"policy"|"elearning",
  eli_or_ecli:     str,
  article:         str|null,
  paragraph:       str|null,
  lid:             str|null,
  onderdeel:       str|null,
  sub:             str|null,
  hierarchy_path:  str,
  effective_date:  date,
  valid_from:      date,
  valid_to:        date|null,
  tax_year:        int|null,
  classification:  "public"|"internal"|"fiod",
  parent_chunk_id: str,
  score:           float   # post-rerank score from Cohere
}
```

Module 3's citation verifier uses `eli_or_ecli + hierarchy_path` as the canonical anchor for regex-based citation grounding checks.

---

## 2.5 Latency Budget

**Handoff to Test agent:** Retrieval-quality metrics exposed at the following K values:
- **Recall@100** (pre-fusion, per retriever arm) — detects if either BM25 or k-NN is failing to surface ground-truth chunks before fusion.
- **nDCG@10** (post-RRF, pre-rerank) — measures RRF ranking quality.
- **MRR** (post-rerank, top-8) — measures whether the single best chunk is surfaced first; most relevant for citation-exact use cases.
- **Recall@8** (post-rerank) — the metric that directly predicts LLM faithfulness.

| Step | p50 | p95 | Notes |
|---|---|---|---|
| Redis semantic cache lookup | 5 ms | 10 ms | SHA-256 key + cosine check on 0.97 threshold |
| Cache miss — query embed (Bedrock) | 40 ms | 70 ms | `embed-multilingual-v3`, single 1024-d vector |
| OpenSearch hybrid query (BM25 + k-NN) | 60 ms | 110 ms | 12 shards, ef_search=128, FP16 vectors in MMAP |
| RRF fusion (in-pipeline) | 5 ms | 10 ms | Server-side, no network hop |
| Cohere rerank-v3-5:0 (Bedrock, top-60→8) | 150 ms | 280 ms | Confirmed Bedrock p95; fallback: skip rerank, return RRF top-8 |
| Parent body `mget` (8 docs) | 10 ms | 20 ms | Single-round-trip multi-get |
| LLM first-token (Haiku 4.5, ~2k ctx) | 350 ms | 600 ms | Cross-region inference profile; TTFT only |
| **End-to-end (cache miss, p95)** | **620 ms** | **1,100 ms** | **Under 1,500 ms budget with 400 ms margin** |

The p95 stack sum is 1,100 ms, leaving 400 ms margin against the 1,500 ms TTFT hard limit. The single largest risk is the Cohere rerank Bedrock call at p95 280 ms; the fallback rule (if rerank wall-clock > 300 ms, return RRF top-8 directly) keeps p99 under budget. This fallback is implemented as a `asyncio.wait_for` timeout wrapper around the Bedrock `rerank` call, not a circuit breaker, to avoid state synchronisation overhead.

---

## Domain Review Findings

### Citation Format (Check 1)

- The single field `eli_or_ecli` in Section 1.2 conflates two structurally incompatible identifier namespaces. ELI has the form `ELI/wet/IB2001/artikel/3.114` (a hierarchical path); ECLI has the form `ECLI:NL:HR:2021:1234` (a colon-delimited authority string). A consumer performing an exact BM25 lookup on `eli_or_ecli` with an ELI string will never collide with an ECLI string, but the single-field design blocks type-safe routing: code that wants "all legislation chunks citing this ELI" must parse the value to distinguish it from case-law chunks. Flag for report-compiler: split into `eli` (keyword, nullable) and `ecli` (keyword, nullable) in Section 1.2 schema table and propagate the change through the pseudo-code in Section 1.1 (`split_case_law` already uses `doc["ecli"]` correctly; `split_legislation` already uses `doc["eli"]` correctly — only the stored field name needs splitting).

### Hierarchy Depth (Check 2)

- Section 1.1 pseudo-code populates `lid`, `onderdeel`, and `sub` fields and builds `hierarchy_path` as `eli/art{article}/lid{lid}/{onderdeel}` — `sub` is parsed but omitted from the path string. For `Artikel 3.114, lid 2, onderdeel a, sub 3°` the path would render as `eli/art3.114/lid2/a`, losing the sub-level. The citation verifier in Module 3 uses `hierarchy_path` as the canonical anchor; a citation to `sub 3°` would fail the anchor check. Flag: append `/{sub}` to `hierarchy_path` construction when `sub` is non-null.
- The regex list `LEGISLATION_HEADERS` (Section 1.1) covers `^\d+°` for sub-levels. This pattern will also match standalone numbered list items that are not fiscal sub-levels (e.g., numbered footnotes in Memories of Explanation). A narrower pattern such as `^\d+°\s` (trailing whitespace required) reduces false splits.

### Temporal Validity (Check 3)

- `tax_year` (short integer) and `valid_from`/`valid_to` are both present in the schema (Section 1.2) and surfaced in the retrieval filter DSL (Section 2.1). This is handled correctly. The filter DSL in Section 2.1 range-filters on `valid_from`/`valid_to` at query time, which is the right mechanism for inspector-scoped retrieval.

### Superseded / Consolidated Versions (Check 4)

- `superseded_by` is present in the schema (Section 1.2) as a keyword field holding the ELI/ECLI of the replacing document. The schema models the one-directional pointer (old -> new). However, there is no `effective_until` alias or display label — `valid_to` serves this purpose but is a date, not a document reference. More critically, the retrieval DSL in Section 2.1 does not filter out superseded chunks by default: a query with `query_date=2024-01-01` will correctly exclude chunks where `valid_to < 2024-01-01`, but a chunk that is superseded mid-year (e.g., `valid_to=2024-06-30`) will still appear for queries dated before the supersession. This is correct behaviour only if inspectors are always aware they may be seeing a pre-supersession version. The generation prompt (Module 3) should receive the `superseded_by` field and surface a disclosure when a cited chunk has a non-null `superseded_by` value. Flag for Module 3 compiler.

### FIOD Classification (Check 5)

- The schema defines `classification` as `public | internal | fiod` (Section 1.2), with FIOD as a distinct ordinal level above `internal`. The DLS rule example (Section 1.4) correctly restricts helpdesk to `["public", "internal"]` — this example should read `["public"]` only for helpdesk, since `internal` is inspector/legal-level access. The DLS code comment says "helpdesk — sees public and internal only" but the Module 4 role matrix assigns helpdesk to `classification: ["public"]` only. The inconsistency between the Section 1.4 DLS example and the Module 4 role matrix must be resolved; the Module 4 matrix is the authoritative source.

### Multilinguality (Check 6)

- Cohere `embed-multilingual-v3` via Bedrock is chosen (Section 2.2). The module explicitly verifies NL fiscal terminology separation and notes AWS Bedrock residency for data sovereignty. The BM25 arm uses the `dutch` analyzer (Section 1.3 index mapping). This combination is adequate for NL-primary + EN EU-directive content. No gaps identified.

### Legal Counsel Role (Check 7)

- The retrieval layer has no legal-counsel-specific configuration. In Module 4's role matrix, `legal` and `inspector` share identical classification ceilings (`["public","internal"]`) and identical DLS roles (`role_legal` vs `role_inspector` are separate IAM ARNs but identical filter scope). The retrieval module itself (Modules 1-2) does not need to change for this, but there is no corpus tag distinguishing privileged legal-counsel memos (e.g., attorney-opinion documents) from standard internal policy documents within the `internal` classification tier. Both inspector and legal counsel would retrieve the same privileged memos. Flag: consider a `subcategory` field (e.g., `legal_privilege`) within the `internal` tier to enable finer DLS scoping for privileged documents, without requiring a fourth classification level.

### Cache Poisoning / Tax-Year Ambiguity (Check 8)

- This concern is addressed in Module 4 (Section 2.3), not in Modules 1-2. The retrieval module correctly propagates `tax_year` in chunk metadata and in the query filter. No gap in Modules 1-2 for this check; see Module 4 findings.
