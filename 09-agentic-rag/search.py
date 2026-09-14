"""
A plain single-shot search tool -- identical to 01/02's retrieval pipeline,
just exposed as a minimal reusable tool. What's different about this
project isn't the search code, it's WHO calls it and how many times: 09's
README compares one call (single-shot RAG) against several calls chosen
adaptively based on what earlier results were missing (agentic RAG).

Usage:
    python search.py "query" --top-k 3
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    load_dotenv(dotenv_path=HERE.parent / ".env")
    client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])

    chunks = load_and_chunk_corpus()
    chunk_vecs = embed_chunks(client, chunks)

    query_result = client.embed([args.query], model=EMBED_MODEL, input_type="query")
    query_vec = np.array(query_result.embeddings[0], dtype=np.float32)

    chunk_norm = chunk_vecs / np.linalg.norm(chunk_vecs, axis=1, keepdims=True)
    query_norm = query_vec / np.linalg.norm(query_vec)
    scores = chunk_norm @ query_norm
    top = np.argsort(-scores)[:args.top_k]

    print(f"=== RESULTS for {args.query!r} ===\n")
    for rank, i in enumerate(top, 1):
        print(f"[{rank}] (source: {chunks[i]['doc']}, score: {scores[i]:.3f})")
        print(chunks[i]["text"])
        print()


if __name__ == "__main__":
    main()
