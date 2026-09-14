"""
Hybrid search: BM25 (keyword) + vector search (semantic), combined with
Reciprocal Rank Fusion.

Vector search finds meaning; it's weak on exact tokens that don't carry
much semantic content of their own -- product codes, error codes, model
numbers. BM25 (the classic keyword-search ranking function used by search
engines for decades) is the reverse: exact-term matching, weak on synonyms
and paraphrase. Hybrid search runs both and merges the rankings, so a query
gets whichever kind of match actually helps.

Usage:
    python rag.py "query" --mode vector
    python rag.py "query" --mode bm25
    python rag.py "query" --mode hybrid
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
from rank_bm25 import BM25Okapi

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

HERE = Path(__file__).parent
DOCS_DIR = HERE / "data" / "docs"
CACHE_FILE = HERE / "data" / "cache" / "embeddings.json"
EMBED_MODEL = "voyage-3.5"
CHUNK_SIZE = 500
RRF_K = 60  # standard RRF damping constant; de-emphasizes rank differences deep in a list


def chunk_recursive(text: str, doc_name: str, chunk_size=CHUNK_SIZE):
    """Paragraph-first structure-aware chunking, same as 02/03."""
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


def tokenize(text: str):
    """Lowercase, keep hyphenated alphanumeric tokens intact (so "ERR-4471"
    stays one token instead of splitting into "err" and "4471") -- that's
    exactly the kind of exact-match token BM25 is good at and embeddings
    tend to blur.
    """
    return re.findall(r"[a-z0-9][a-z0-9\-]*", text.lower())


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


def vector_ranking(query_vec, chunk_vecs):
    chunk_norm = chunk_vecs / np.linalg.norm(chunk_vecs, axis=1, keepdims=True)
    query_norm = query_vec / np.linalg.norm(query_vec)
    scores = chunk_norm @ query_norm
    ranked = np.argsort(-scores)
    return [(int(i), float(scores[i])) for i in ranked]


def bm25_ranking(bm25, query):
    scores = bm25.get_scores(tokenize(query))
    ranked = np.argsort(-scores)
    return [(int(i), float(scores[i])) for i in ranked]


def reciprocal_rank_fusion(rankings, k=RRF_K):
    """Combine multiple ranked lists into one, using only each item's RANK
    in each list (not its raw score) -- this sidesteps the problem that
    BM25 scores and cosine similarities live on completely different,
    incomparable scales. score(doc) = sum over rankers of 1 / (k + rank).
    """
    fused = {}
    for ranking in rankings:
        for rank, (idx, _score) in enumerate(ranking):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda kv: -kv[1])


def format_handoff(query, mode, chunks, results, top_k):
    lines = [f"=== RETRIEVED CONTEXT (mode: {mode}) ===\n"]
    for i, (idx, score) in enumerate(results[:top_k], 1):
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
    parser.add_argument("--mode", choices=["vector", "bm25", "hybrid"], default="hybrid")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    chunks = load_and_chunk_corpus()

    vector_results = None
    bm25_results = None

    if args.mode in ("vector", "hybrid"):
        load_dotenv(dotenv_path=HERE.parent / ".env")
        client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
        chunk_vecs = embed_chunks(client, chunks)
        query_result = client.embed([args.query], model=EMBED_MODEL, input_type="query")
        query_vec = np.array(query_result.embeddings[0], dtype=np.float32)
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
