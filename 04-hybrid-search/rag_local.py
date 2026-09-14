"""
Same hybrid BM25 + vector search as rag.py, but the vector half uses a
local model (BAAI/bge-small-en-v1.5, run on-device via fastembed) instead
of Voyage's API. BM25 is untouched either way -- it's pure keyword
statistics and never calls an embedding model at all.

Usage:
    python rag_local.py "query" --mode vector
    python rag_local.py "query" --mode bm25
    python rag_local.py "query" --mode hybrid
"""

import argparse
import hashlib
import json
import sys

import numpy as np
from fastembed import TextEmbedding
from rank_bm25 import BM25Okapi

from rag import (
    HERE,
    load_and_chunk_corpus,
    tokenize,
    vector_ranking,
    bm25_ranking,
    reciprocal_rank_fusion,
    format_handoff,
)

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

CACHE_FILE_LOCAL = HERE / "data" / "cache" / "embeddings_local.json"
EMBED_MODEL_LOCAL = "BAAI/bge-small-en-v1.5"


def embed_chunks_local(model, chunks):
    payload = json.dumps({"chunks": [c["text"] for c in chunks], "model": EMBED_MODEL_LOCAL}, sort_keys=True)
    key = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    if CACHE_FILE_LOCAL.exists():
        cached = json.loads(CACHE_FILE_LOCAL.read_text(encoding="utf-8"))
        if cached.get("key") == key:
            return np.array(cached["embeddings"], dtype=np.float32)

    embeddings = np.array(list(model.passage_embed([c["text"] for c in chunks])), dtype=np.float32)
    CACHE_FILE_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE_LOCAL.write_text(json.dumps({"key": key, "embeddings": embeddings.tolist()}), encoding="utf-8")
    return embeddings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--mode", choices=["vector", "bm25", "hybrid"], default="hybrid")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    chunks = load_and_chunk_corpus()

    vector_results = None
    bm25_results = None

    if args.mode in ("vector", "hybrid"):
        model = TextEmbedding(model_name=EMBED_MODEL_LOCAL)
        chunk_vecs = embed_chunks_local(model, chunks)
        query_vec = np.array(list(model.query_embed([args.query]))[0], dtype=np.float32)
        vector_results = vector_ranking(query_vec, chunk_vecs)

    if args.mode in ("bm25", "hybrid"):
        bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
        bm25_results = bm25_ranking(bm25, args.query)

    if args.mode == "vector":
        results = vector_results
    elif args.mode == "bm25":
        results = bm25_results
    else:
        results = reciprocal_rank_fusion([vector_results, bm25_results])

    print(format_handoff(args.query, args.mode, chunks, results, args.top_k))


if __name__ == "__main__":
    main()
