"""
Same naive RAG pipeline as rag.py, but the embedding step is swapped from
Voyage's hosted API to a local open-source model (BAAI/bge-small-en-v1.5,
384 dims, run on-device via fastembed/ONNX). No API key, no network call
at query time, nothing leaves your machine.

Run both on the same query and diff the retrieved chunks/scores:
    python rag.py "How long is the warranty on the Hub 3?"
    python rag_local.py "How long is the warranty on the Hub 3?"
"""

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
from fastembed import TextEmbedding

from rag import (
    HERE,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    load_and_chunk_corpus,
    cosine_similarity,
    format_handoff,
)
import argparse
import hashlib

CACHE_FILE_LOCAL = HERE / "data" / "cache" / "embeddings_local.json"
EMBED_MODEL_LOCAL = "BAAI/bge-small-en-v1.5"


def _cache_key(chunks):
    payload = json.dumps(
        {"chunks": [c["text"] for c in chunks], "model": EMBED_MODEL_LOCAL,
         "chunk_size": CHUNK_SIZE, "overlap": CHUNK_OVERLAP},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def embed_chunks_local(model, chunks):
    key = _cache_key(chunks)
    if CACHE_FILE_LOCAL.exists():
        cached = json.loads(CACHE_FILE_LOCAL.read_text(encoding="utf-8"))
        if cached.get("key") == key:
            return np.array(cached["embeddings"], dtype=np.float32)

    texts = [c["text"] for c in chunks]
    embeddings = np.array(list(model.passage_embed(texts)), dtype=np.float32)

    CACHE_FILE_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE_LOCAL.write_text(
        json.dumps({"key": key, "embeddings": embeddings.tolist()}),
        encoding="utf-8",
    )
    return embeddings


def embed_query_local(model, query: str):
    return np.array(list(model.query_embed([query]))[0], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="The question to retrieve context for")
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve")
    args = parser.parse_args()

    model = TextEmbedding(model_name=EMBED_MODEL_LOCAL)

    chunks = load_and_chunk_corpus()
    chunk_vecs = embed_chunks_local(model, chunks)
    query_vec = embed_query_local(model, args.query)
    scores = cosine_similarity(query_vec, chunk_vecs)

    top_indices = np.argsort(-scores)[:args.top_k]
    results = [(chunks[i], float(scores[i])) for i in top_indices]

    print(format_handoff(args.query, results))


if __name__ == "__main__":
    main()
