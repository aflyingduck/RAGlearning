"""
Same brute-force-vs-HNSW comparison as real_corpus_demo.py, but embeddings
come from a local model (BAAI/bge-small-en-v1.5, run on-device via
fastembed) instead of Voyage's API. No API key, no network call at query
time. The ANN-vs-brute-force question this project is about is orthogonal
to which embedding model produced the vectors -- this just swaps that one
piece.

Usage:
    python real_corpus_demo_local.py "your question"
"""

import argparse
import hashlib
import json
import sys
import time

import numpy as np
from fastembed import TextEmbedding

from real_corpus_demo import HERE, load_and_chunk_corpus, brute_force_search, hnsw_search

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
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    model = TextEmbedding(model_name=EMBED_MODEL_LOCAL)

    chunks = load_and_chunk_corpus()
    chunk_vecs = embed_chunks_local(model, chunks)
    query_vec = np.array(list(model.query_embed([args.query]))[0], dtype=np.float32)

    bf_start = time.perf_counter()
    bf_top, bf_scores = brute_force_search(query_vec, chunk_vecs, args.top_k)
    bf_time = time.perf_counter() - bf_start

    hnsw_start = time.perf_counter()
    hnsw_top, hnsw_scores = hnsw_search(query_vec, chunk_vecs, args.top_k)
    hnsw_time = time.perf_counter() - hnsw_start

    print(f"Corpus size: {len(chunks)} chunks\n")
    print(f"Brute-force: {bf_time*1000:.3f} ms")
    for i, s in zip(bf_top, bf_scores):
        print(f"  [{i}] {s:.3f}  {chunks[i]['doc']}: {chunks[i]['text'][:70]!r}")

    print(f"\nHNSW (approximate): {hnsw_time*1000:.3f} ms")
    for i, s in zip(hnsw_top, hnsw_scores):
        print(f"  [{i}] {s:.3f}  {chunks[i]['doc']}: {chunks[i]['text'][:70]!r}")

    same = set(bf_top.tolist()) == set(hnsw_top.tolist())
    print(f"\nSame top-{args.top_k} chunks: {same}")


if __name__ == "__main__":
    main()
