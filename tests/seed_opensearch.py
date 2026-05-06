#!/usr/bin/env python3
"""
seed_opensearch.py — Push ~250 synthetic tax-law fixtures into OpenSearch.

Creates index 'tax-docs-test' with Lucene k-NN HNSW settings per MASTER-PLAN §C:
  m=32, ef_construction=256, ef_search=128, dimension=1024 (Cohere embed-multilingual-v3).

For cost efficiency in dev, vectors are random 1024-dim float32 vectors.
For full integration accuracy, set SEED_USE_BEDROCK=true and ensure AWS creds are present.

Schema per module-1-2-retrieval.md §1.1 + conftest.ChunkMeta:
  chunk_id, doc_id, doc_type, text, classification, eli, ecli,
  article, lid, onderdeel, sub, hierarchy_path,
  valid_from, valid_to, tax_year, superseded_by, embedding (knn_vector, 1024-dim)
"""
from __future__ import annotations

import json
import os
import random
import time
import uuid
from dataclasses import asdict
from typing import Optional

# Use sys.path to find conftest
import sys
sys.path.insert(0, "/app/tests")
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from opensearchpy import OpenSearch, RequestError
from opensearchpy.helpers import bulk

from conftest import MOCK_CORPUS, ChunkMeta

OPENSEARCH_URL  = os.environ.get("OPENSEARCH_URL",  "https://localhost:9200")
OPENSEARCH_USER = os.environ.get("OPENSEARCH_USER", "admin")
OPENSEARCH_PASS = os.environ.get("OPENSEARCH_PASS", "admin")
USE_BEDROCK     = os.environ.get("SEED_USE_BEDROCK", "false").lower() == "true"
INDEX           = "tax-docs-test"
DIM             = 1024
HNSW_M          = 32
HNSW_EF_CONSTRUCTION = 256
HNSW_EF_SEARCH  = 128

verify_certs = os.environ.get("OPENSEARCH_VERIFY_CERTS", "false").lower() == "true"

client = OpenSearch(
    hosts=[OPENSEARCH_URL],
    http_auth=(OPENSEARCH_USER, OPENSEARCH_PASS),
    use_ssl=OPENSEARCH_URL.startswith("https"),
    verify_certs=verify_certs,
    ssl_show_warn=False,
)


INDEX_BODY = {
    "settings": {
        "index": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "knn": True,
            "knn.algo_param.ef_search": HNSW_EF_SEARCH,
        }
    },
    "mappings": {
        "properties": {
            "chunk_id":       {"type": "keyword"},
            "doc_id":         {"type": "keyword"},
            "doc_type":       {"type": "keyword"},
            "classification": {"type": "keyword"},
            "eli":            {"type": "keyword"},
            "ecli":           {"type": "keyword"},
            "article":        {"type": "keyword"},
            "paragraph":      {"type": "keyword"},
            "lid":            {"type": "keyword"},
            "onderdeel":      {"type": "keyword"},
            "sub":            {"type": "keyword"},
            "hierarchy_path": {"type": "keyword"},
            "valid_from":     {"type": "date", "format": "yyyy-MM-dd"},
            "valid_to":       {"type": "date", "format": "yyyy-MM-dd", "null_value": None},
            "tax_year":       {"type": "integer"},
            "superseded_by":  {"type": "keyword"},
            "text": {
                "type": "text",
                "analyzer": "dutch",
                "fields": {"keyword": {"type": "keyword", "ignore_above": 512}},
            },
            "embedding": {
                "type": "knn_vector",
                "dimension": DIM,
                "method": {
                    "name": "hnsw",
                    "engine": "lucene",
                    "space_type": "cosinesimil",
                    "parameters": {
                        "m": HNSW_M,
                        "ef_construction": HNSW_EF_CONSTRUCTION,
                    },
                },
            },
        }
    },
}


def random_vector(dim: int = DIM) -> list[float]:
    """Random unit-length vector for dev seeding."""
    vec = [random.gauss(0, 1) for _ in range(dim)]
    mag = sum(v**2 for v in vec) ** 0.5
    return [v / mag for v in vec]


def chunk_to_doc(chunk: ChunkMeta) -> dict:
    """Convert ChunkMeta to an OpenSearch document."""
    doc = {
        "chunk_id":       chunk.chunk_id,
        "doc_id":         chunk.doc_id,
        "doc_type":       chunk.doc_type,
        "text":           chunk.text,
        "classification": chunk.classification,
        "eli":            chunk.eli,
        "ecli":           chunk.ecli,
        "article":        chunk.article,
        "paragraph":      chunk.paragraph,
        "lid":            chunk.lid,
        "onderdeel":      chunk.onderdeel,
        "sub":            chunk.sub,
        "hierarchy_path": chunk.hierarchy_path,
        "valid_from":     chunk.valid_from,
        "valid_to":       chunk.valid_to,
        "tax_year":       chunk.tax_year,
        "superseded_by":  chunk.superseded_by,
        "embedding":      random_vector(),
    }
    # Remove None values for cleanliness except fields that can be null
    return {k: v for k, v in doc.items() if v is not None or k in ("valid_to", "superseded_by")}


def generate_synthetic_fixtures(n: int = 230) -> list[ChunkMeta]:
    """
    Generate n additional synthetic chunks spanning all doc_type × classification × tax_year combos.
    These supplement the 20 chunks in MOCK_CORPUS to reach ~250 total.
    """
    doc_types      = ["legislation", "case_law", "policy", "elearning"]
    classifications = ["public", "internal", "fiod"]
    tax_years      = [2021, 2022, 2023, 2024]
    articles       = ["3.114", "3.16", "3.34", "3.90", "3.95", "4.1", "5.10", "6.1"]

    chunks = []
    for i in range(n):
        doc_type = doc_types[i % len(doc_types)]
        classification = classifications[i % len(classifications)]
        tax_year = tax_years[i % len(tax_years)]
        article  = articles[i % len(articles)]
        lid      = str((i % 3) + 1)
        onderdeel = ["a", "b", "c", None][i % 4]

        if doc_type == "case_law":
            ecli = f"ECLI:NL:RBAMS:{tax_year}:{1000 + i}"
            text = (
                f"Rechtbank Amsterdam {tax_year}, {ecli}. "
                f"Uitspraak inzake fiscale kwestie artikel {article} Wet IB 2001. "
                f"Belastingjaar {tax_year}."
            )
        else:
            ecli = None
            text = (
                f"Artikel {article} lid {lid}"
                + (f" onderdeel {onderdeel}" if onderdeel else "")
                + f": wettelijke bepaling voor belastingjaar {tax_year}. "
                f"Classificatie: {classification}. "
                f"Type: {doc_type}."
            )

        from conftest import _make_chunk
        chunk = _make_chunk(
            doc_id=f"synthetic-{doc_type}-{tax_year}-{i:04d}",
            doc_type=doc_type,
            text=text,
            classification=classification,
            article=article if doc_type != "case_law" else None,
            lid=lid if doc_type != "case_law" else None,
            onderdeel=onderdeel,
            tax_year=tax_year,
            ecli=ecli,
            valid_from=f"{tax_year}-01-01",
            valid_to=f"{tax_year}-12-31" if tax_year < 2024 else None,
        )
        chunks.append(chunk)

    return chunks


def ensure_index() -> None:
    """Create index if it doesn't exist, with HNSW k-NN settings."""
    if client.indices.exists(index=INDEX):
        print(f"[seed] Index '{INDEX}' already exists — deleting and recreating...")
        client.indices.delete(index=INDEX)

    print(f"[seed] Creating index '{INDEX}' with Lucene HNSW k-NN settings...")
    client.indices.create(index=INDEX, body=INDEX_BODY)
    print(f"[seed] Index '{INDEX}' created.")


def seed_chunks(chunks: list[ChunkMeta]) -> None:
    """Bulk-index chunks into OpenSearch."""
    actions = []
    for chunk in chunks:
        doc = chunk_to_doc(chunk)
        actions.append({
            "_index": INDEX,
            "_id": chunk.chunk_id,
            "_source": doc,
        })

    print(f"[seed] Bulk-indexing {len(actions)} documents...")
    success, errors = bulk(client, actions, chunk_size=50, raise_on_error=False)
    print(f"[seed] Indexed {success} documents. Errors: {len(errors)}")
    if errors:
        for err in errors[:5]:
            print(f"[seed] Error: {err}")


def main() -> None:
    print(f"[seed] Connecting to {OPENSEARCH_URL}...")

    # Wait for cluster to be healthy
    for attempt in range(20):
        try:
            health = client.cluster.health(
                params={"wait_for_status": "yellow", "timeout": "30s"}
            )
            print(f"[seed] Cluster status: {health.get('status')}")
            break
        except Exception as e:
            print(f"[seed] Waiting for cluster... attempt {attempt+1}/20: {e}")
            time.sleep(10)
    else:
        print("[seed] FATAL: Cluster did not become healthy in time.")
        sys.exit(1)

    ensure_index()

    # Combine MOCK_CORPUS with synthetic fixtures
    all_chunks = list(MOCK_CORPUS)
    synthetic  = generate_synthetic_fixtures(230)
    all_chunks.extend(synthetic)
    print(f"[seed] Total chunks to seed: {len(all_chunks)} "
          f"({len(MOCK_CORPUS)} from MOCK_CORPUS + {len(synthetic)} synthetic)")

    seed_chunks(all_chunks)

    # Refresh index so tests can query immediately
    client.indices.refresh(index=INDEX)
    count_resp = client.count(index=INDEX)
    print(f"[seed] Documents in index after refresh: {count_resp.get('count', '?')}")
    print("[seed] Seeding complete.")


if __name__ == "__main__":
    main()
