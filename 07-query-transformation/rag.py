"""
Query transformation: rewrite or expand the query BEFORE searching, instead
of embedding it as-typed.

Two techniques:
- Multi-query: split one question into several narrower sub-questions,
  retrieve for each separately, and fuse the rankings (same RRF as
  04-hybrid-search). Helps compound questions, where one query embedding
  has to represent multiple distinct facts at once and ends up
  representing none of them precisely.
- HyDE (Hypothetical Document Embeddings): generate a fake but plausible
  ANSWER to the query, and embed that instead of the query itself. The
  idea: a real answer chunk is more likely to resemble another
  answer-shaped piece of text than it is to resemble a question. Note the
  hypothetical doc is embedded with input_type="document", not "query" --
  that's the whole mechanism.

The sub-queries and the hypothetical document are both generation tasks,
so per this curriculum's pattern they're written by Claude in-chat, not
produced by an API call inside this script -- pass them in via CLI flags.

Usage:
    python rag.py "original query" --mode baseline
    python rag.py "original query" --mode multiquery --sub-queries '["sub q 1", "sub q 2"]'
    python rag.py "original query" --mode hyde --hyde-doc "a plausible hypothetical answer..."
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
CHUNK_SIZE = 500
RRF_K = 60


def chunk_recursive(text, doc_name, chunk_size=CHUNK_SIZE):
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


def vector_ranking(query_vec, chunk_vecs):
    chunk_norm = chunk_vecs / np.linalg.norm(chunk_vecs, axis=1, keepdims=True)
    query_norm = query_vec / np.linalg.norm(query_vec)
    scores = chunk_norm @ query_norm
    ranked = np.argsort(-scores)
    return [(int(i), float(scores[i])) for i in ranked]


def rrf_fuse(rankings, k=RRF_K):
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
    parser.add_argument("--mode", choices=["baseline", "multiquery", "hyde"], default="baseline")
    parser.add_argument("--sub-queries", type=str, help="JSON list of sub-question strings")
    parser.add_argument("--hyde-doc", type=str, help="Hypothetical answer text")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    load_dotenv(dotenv_path=HERE.parent / ".env")
    client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])

    chunks = load_and_chunk_corpus()
    chunk_vecs = embed_chunks(client, chunks)

    if args.mode == "baseline":
        qres = client.embed([args.query], model=EMBED_MODEL, input_type="query")
        qvec = np.array(qres.embeddings[0], dtype=np.float32)
        results = vector_ranking(qvec, chunk_vecs)

    elif args.mode == "multiquery":
        if not args.sub_queries:
            parser.error("--mode multiquery requires --sub-queries")
        sub_queries = json.loads(args.sub_queries)
        qres = client.embed(sub_queries, model=EMBED_MODEL, input_type="query")
        rankings = [vector_ranking(np.array(v, dtype=np.float32), chunk_vecs) for v in qres.embeddings]
        results = rrf_fuse(rankings)

    else:  # hyde
        if not args.hyde_doc:
            parser.error("--mode hyde requires --hyde-doc")
        # Embedded as input_type="document" -- deliberately, since a
        # hypothetical ANSWER is document-shaped text, not query-shaped.
        qres = client.embed([args.hyde_doc], model=EMBED_MODEL, input_type="document")
        qvec = np.array(qres.embeddings[0], dtype=np.float32)
        results = vector_ranking(qvec, chunk_vecs)

    print(format_handoff(args.query, args.mode, chunks, results, args.top_k))


if __name__ == "__main__":
    main()
