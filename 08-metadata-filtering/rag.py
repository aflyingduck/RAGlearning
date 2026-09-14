"""
Metadata filtering: attach structured fields to each chunk (here, which
Hub generation a doc applies to) and let a query restrict the candidate
set BEFORE vector search runs, instead of trusting embedding similarity
alone to keep unrelated-but-similar content out.

This corpus now documents two hardware generations (Hub 2 and Hub 3) that
share a lot of vocabulary -- "factory reset," "device limit," "warranty" --
but have different specific answers. Vector search has no way to know
which generation a query is about unless the query happens to say so in
words that embed distinctly; metadata filtering makes that explicit and
guaranteed instead of hoping the embedding sorts it out.

Usage:
    python rag.py "query" --top-k 3                     # no filter
    python rag.py "query" --top-k 3 --model hub2         # only hub2-tagged chunks
    python rag.py "query" --top-k 3 --model hub3
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
METADATA_FILE = HERE / "data" / "metadata.json"
CACHE_FILE = HERE / "data" / "cache" / "embeddings.json"
EMBED_MODEL = "voyage-3.5"
CHUNK_SIZE = 500


def chunk_recursive(text, doc_name, metadata, chunk_size=CHUNK_SIZE):
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
                chunks.append({"doc": doc_name, "text": current, "meta": metadata})
                current = ""
            for sub in pack_sentences(para):
                chunks.append({"doc": doc_name, "text": sub, "meta": metadata})
            continue
        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) > chunk_size and current:
            chunks.append({"doc": doc_name, "text": current, "meta": metadata})
            current = para
        else:
            current = candidate
    if current:
        chunks.append({"doc": doc_name, "text": current, "meta": metadata})
    return chunks


def load_and_chunk_corpus():
    metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    all_chunks = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        doc_meta = metadata.get(path.name, {})
        all_chunks.extend(chunk_recursive(path.read_text(encoding="utf-8"), path.name, doc_meta))
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


def vector_search(query_vec, chunk_vecs, candidate_indices, k):
    """Search only among candidate_indices -- the metadata-filtered subset,
    if a filter was given, or every chunk if not. Filtering happens BEFORE
    scoring, not after: a filtered-out chunk never gets a chance to
    outscore an in-filter one, unlike re-ranking or post-hoc removal.
    """
    sub_vecs = chunk_vecs[candidate_indices]
    sub_norm = sub_vecs / np.linalg.norm(sub_vecs, axis=1, keepdims=True)
    query_norm = query_vec / np.linalg.norm(query_vec)
    scores = sub_norm @ query_norm
    order = np.argsort(-scores)[:k]
    return [(candidate_indices[i], float(scores[i])) for i in order]


def format_handoff(query, filter_desc, chunks, results):
    lines = [f"=== RETRIEVED CONTEXT (filter: {filter_desc}) ===\n"]
    for i, (idx, score) in enumerate(results, 1):
        chunk = chunks[idx]
        lines.append(f"[{i}] (source: {chunk['doc']}, model: {chunk['meta'].get('model', '?')}, score: {score:.4f})")
        lines.append(chunk["text"])
        lines.append("")
    lines.append("=== QUERY ===")
    lines.append(query)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--model", choices=["hub2", "hub3"], help="Restrict results to chunks tagged for this Hub generation")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    load_dotenv(dotenv_path=HERE.parent / ".env")
    client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])

    chunks = load_and_chunk_corpus()
    chunk_vecs = embed_chunks(client, chunks)

    if args.model:
        candidate_indices = [i for i, c in enumerate(chunks) if c["meta"].get("model") == args.model]
        filter_desc = f"model={args.model}"
    else:
        candidate_indices = list(range(len(chunks)))
        filter_desc = "none"

    query_result = client.embed([args.query], model=EMBED_MODEL, input_type="query")
    query_vec = np.array(query_result.embeddings[0], dtype=np.float32)

    results = vector_search(query_vec, chunk_vecs, candidate_indices, args.top_k)
    print(format_handoff(args.query, filter_desc, chunks, results))


if __name__ == "__main__":
    main()
