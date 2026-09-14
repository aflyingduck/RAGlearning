"""
Naive RAG pipeline: chunk -> embed -> cosine search -> retrieve.

This script does NOT call an LLM to generate an answer. It stops after
retrieval and prints the retrieved context plus your query in a clearly
delimited block. Paste that block to Claude Code (this same chat) to get
the generated answer. Keeping retrieval and generation as separate, visible
steps is the point of this exercise -- RAG quality lives almost entirely in
the retrieval step, and that's invisible if an LLM call hides it from you.

Usage:
    python rag.py "How long is the warranty on the Hub 3?"
    python rag.py "How long is the warranty on the Hub 3?" --top-k 5
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

# Windows terminals default stdout/stderr to cp1252, which mangles the
# em-dashes and curly quotes in the source docs. Force UTF-8 so retrieved
# text prints correctly regardless of platform.
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import voyageai
from dotenv import load_dotenv

HERE = Path(__file__).parent
DOCS_DIR = HERE / "data" / "docs"
CACHE_FILE = HERE / "data" / "cache" / "embeddings.json"
EMBED_MODEL = "voyage-3.5"
CHUNK_SIZE = 500      # characters per chunk
CHUNK_OVERLAP = 50    # characters shared between consecutive chunks


def chunk_text(text: str, doc_name: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Fixed-size character chunking with overlap.

    This is the simplest possible chunking strategy: it knows nothing about
    sentence or paragraph boundaries, so it will sometimes cut a fact in
    half across two chunks. That's intentional here -- project 02 fixes
    this. The overlap exists to reduce (not eliminate) boundary damage.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({"doc": doc_name, "text": chunk})
        start += chunk_size - overlap
    return chunks


def load_and_chunk_corpus():
    all_chunks = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        all_chunks.extend(chunk_text(text, doc_name=path.name))
    return all_chunks


def _cache_key(chunks):
    # Hash chunk contents + model + chunking params so the cache invalidates
    # itself automatically if you edit a doc or change CHUNK_SIZE.
    payload = json.dumps(
        {"chunks": [c["text"] for c in chunks], "model": EMBED_MODEL,
         "chunk_size": CHUNK_SIZE, "overlap": CHUNK_OVERLAP},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def embed_chunks(client, chunks):
    """Embed all chunks, using a disk cache keyed on content + params.

    Voyage (like most embedding APIs) distinguishes "document" embeddings
    (things you index) from "query" embeddings (things you search with).
    Some models optimize these differently even though both come back as
    same-dimensional vectors -- always match input_type to how the text is
    being used.
    """
    key = _cache_key(chunks)
    if CACHE_FILE.exists():
        cached = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        if cached.get("key") == key:
            return np.array(cached["embeddings"], dtype=np.float32)

    texts = [c["text"] for c in chunks]
    result = client.embed(texts, model=EMBED_MODEL, input_type="document")
    embeddings = np.array(result.embeddings, dtype=np.float32)

    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(
        json.dumps({"key": key, "embeddings": embeddings.tolist()}),
        encoding="utf-8",
    )
    return embeddings


def embed_query(client, query: str):
    result = client.embed([query], model=EMBED_MODEL, input_type="query")
    return np.array(result.embeddings[0], dtype=np.float32)


def cosine_similarity(query_vec, chunk_vecs):
    """Cosine similarity = dot product of normalized vectors.

    Written out with plain numpy (rather than a vector-DB library) so the
    math is visible: this literally is the whole search step in naive RAG.
    """
    query_norm = query_vec / np.linalg.norm(query_vec)
    chunk_norms = chunk_vecs / np.linalg.norm(chunk_vecs, axis=1, keepdims=True)
    return chunk_norms @ query_norm


def format_handoff(query: str, results):
    lines = ["=== RETRIEVED CONTEXT ===\n"]
    for i, (chunk, score) in enumerate(results, 1):
        lines.append(f"[{i}] (source: {chunk['doc']}, similarity: {score:.3f})")
        lines.append(chunk["text"])
        lines.append("")
    lines.append("=== QUERY ===")
    lines.append(query)
    lines.append("")
    lines.append(
        "=== INSTRUCTIONS ===\n"
        "Answer the query using ONLY the retrieved context above. If the "
        "context doesn't contain the answer, say so explicitly rather than "
        "using outside knowledge."
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="The question to retrieve context for")
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve")
    parser.add_argument("--show-scores", action="store_true",
                         help="Also print similarity scores for ALL chunks, not just top-k")
    args = parser.parse_args()

    load_dotenv(dotenv_path=HERE.parent / ".env")
    client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])

    chunks = load_and_chunk_corpus()
    chunk_vecs = embed_chunks(client, chunks)
    query_vec = embed_query(client, args.query)
    scores = cosine_similarity(query_vec, chunk_vecs)

    if args.show_scores:
        print(f"--- all {len(chunks)} chunk scores (sorted) ---", file=sys.stderr)
        for i in np.argsort(-scores):
            print(f"{scores[i]:.3f}  {chunks[i]['doc']}  {chunks[i]['text'][:60]!r}",
                  file=sys.stderr)
        print("", file=sys.stderr)

    top_indices = np.argsort(-scores)[:args.top_k]
    results = [(chunks[i], float(scores[i])) for i in top_indices]

    print(format_handoff(args.query, results))


if __name__ == "__main__":
    main()
