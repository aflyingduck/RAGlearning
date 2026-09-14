"""
Retrieve wide with vector search, then rerank down to the final top-k using
Voyage's rerank API (a cross-encoder: it scores each (query, chunk) PAIR
jointly, instead of comparing two independently-computed embeddings).
Cross-encoders are far more expensive per comparison than embedding cosine
similarity -- too expensive to run against an entire large corpus -- so the
standard pattern is "cheap retrieval gets you a wide candidate set, then an
expensive reranker sorts just those candidates properly."

Usage:
    python rag.py "query" --top-k 3                 # vector top-3, no reranking
    python rag.py "query" --top-k 3 --rerank         # vector top-10, reranked down to 3
    python rag.py "query" --top-k 3 --rerank --retrieve-k 15
"""

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import voyageai
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

HERE = Path(__file__).parent
DOCS_DIR = HERE / "data" / "docs"
CACHE_FILE = HERE / "data" / "cache" / "embeddings.json"
EMBED_MODEL = "voyage-3.5"
RERANK_MODEL = "rerank-2.5"
CHUNK_SIZE = 500


def chunk_recursive(text: str, doc_name: str, chunk_size=CHUNK_SIZE):
    sentence_re = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

    def split_sentences(t):
        t = t.strip()
        return [s.strip() for s in sentence_re.split(t) if s.strip()] if t else []

    def pack_sentences(t):
        chunks, current = [], ""
        for sentence in split_sentences(t):
            candidate = f"{current} {sentence}".strip() if current else sentence
            if len(candidate) > chunk_size and current:
                chunks.append(current)
                current = sentence
            else:
                current = candidate
        if current:
            chunks.append(current)
        return chunks

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, current = [], ""
    for para in paragraphs:
        if len(para) > chunk_size:
            if current:
                chunks.append({"doc": doc_name, "text": current})
                current = ""
            for sub in pack_sentences(para):
                chunks.append({"doc": doc_name, "text": sub})
            continue
        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) > chunk_size and current:
            chunks.append({"doc": doc_name, "text": current})
            current = para
        else:
            current = candidate
    if current:
        chunks.append({"doc": doc_name, "text": current})
    return chunks


def load_and_chunk_corpus():
    all_chunks = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        all_chunks.extend(chunk_recursive(path.read_text(encoding="utf-8"), path.name))
    return all_chunks


def embed_chunks(client, chunks):
    payload = json.dumps({"chunks": [c["text"] for c in chunks], "model": EMBED_MODEL}, sort_keys=True)
    key = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    if CACHE_FILE.exists():
        cached = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        if cached.get("key") == key:
            return np.array(cached["embeddings"], dtype=np.float32)

    result = client.embed([c["text"] for c in chunks], model=EMBED_MODEL, input_type="document")
    embeddings = np.array(result.embeddings, dtype=np.float32)
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps({"key": key, "embeddings": embeddings.tolist()}), encoding="utf-8")
    return embeddings


def vector_search(query_vec, chunk_vecs, k):
    chunk_norm = chunk_vecs / np.linalg.norm(chunk_vecs, axis=1, keepdims=True)
    query_norm = query_vec / np.linalg.norm(query_vec)
    scores = chunk_norm @ query_norm
    ranked = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in ranked]


def rerank(client, query, chunks, candidates, k):
    """candidates: list of (chunk_index, vector_score), already vector-ranked.
    Sends their TEXT to the cross-encoder and returns a new ordering."""
    texts = [chunks[i]["text"] for i, _ in candidates]
    result = client.rerank(query, texts, model=RERANK_MODEL, top_k=k)
    return [(candidates[r.index][0], r.relevance_score) for r in result.results]


def format_handoff(query, mode, chunks, results):
    lines = [f"=== RETRIEVED CONTEXT (mode: {mode}) ===\n"]
    for i, (idx, score) in enumerate(results, 1):
        chunk = chunks[idx]
        lines.append(f"[{i}] (source: {chunk['doc']}, score: {score:.4f})")
        lines.append(chunk["text"])
        lines.append("")
    lines.append("=== QUERY ===")
    lines.append(query)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--retrieve-k", type=int, default=10, help="How many candidates vector search retrieves before reranking")
    parser.add_argument("--rerank", action="store_true")
    args = parser.parse_args()

    load_dotenv(dotenv_path=HERE.parent / ".env")
    client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])

    chunks = load_and_chunk_corpus()
    chunk_vecs = embed_chunks(client, chunks)
    query_result = client.embed([args.query], model=EMBED_MODEL, input_type="query")
    query_vec = np.array(query_result.embeddings[0], dtype=np.float32)

    if args.rerank:
        candidates = vector_search(query_vec, chunk_vecs, args.retrieve_k)
        results = rerank(client, args.query, chunks, candidates, args.top_k)
        mode = f"vector top-{args.retrieve_k} -> reranked to top-{args.top_k}"
    else:
        results = vector_search(query_vec, chunk_vecs, args.top_k)
        mode = f"vector top-{args.top_k} (no reranking)"

    print(format_handoff(args.query, mode, chunks, results))


if __name__ == "__main__":
    main()
