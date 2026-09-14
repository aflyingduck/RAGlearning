"""
Same retrieval pipeline as 01-naive-rag, but with a --strategy flag to swap
the chunker: fixed (01's baseline), sentence, recursive, or semantic. Run
the same query against multiple strategies to see how chunk boundaries
change what gets retrieved.

Usage:
    python rag.py "query" --strategy fixed
    python rag.py "query" --strategy recursive
    python rag.py "query" --strategy semantic --top-k 5
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import voyageai
from dotenv import load_dotenv

from chunkers import NON_SEMANTIC_STRATEGIES, chunk_semantic, split_sentences

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

HERE = Path(__file__).parent
DOCS_DIR = HERE / "data" / "docs"
CACHE_FILE = HERE / "data" / "cache" / "embeddings.json"
EMBED_MODEL = "voyage-3.5"
CHUNK_SIZE = 500


def build_chunks(strategy: str, client):
    """Dispatch to the requested chunker. Semantic chunking needs an
    embed_fn (it embeds sentences to find topic breakpoints), so it's
    wired up separately from the other three, which are pure text
    transforms.
    """
    all_chunks = []
    docs = sorted(DOCS_DIR.glob("*.md"))

    if strategy == "semantic":
        # Batch every sentence from every doc into ONE embed call, rather
        # than one call per doc -- the free tier's 3 req/min limit makes
        # per-document calls expensive fast.
        doc_texts = [(path.name, path.read_text(encoding="utf-8")) for path in docs]
        doc_sentences = [split_sentences(text) for _, text in doc_texts]
        flat_sentences = [s for sents in doc_sentences for s in sents]

        result = client.embed(flat_sentences, model=EMBED_MODEL, input_type="document")
        flat_vecs = np.array(result.embeddings, dtype=np.float32)

        offset = 0
        for (doc_name, text), sents in zip(doc_texts, doc_sentences):
            vecs = flat_vecs[offset:offset + len(sents)]
            offset += len(sents)
            all_chunks.extend(chunk_semantic(text, doc_name, vecs))
        return all_chunks

    chunk_fn = NON_SEMANTIC_STRATEGIES[strategy]
    for path in docs:
        text = path.read_text(encoding="utf-8")
        all_chunks.extend(chunk_fn(text, path.name, chunk_size=CHUNK_SIZE))
    return all_chunks


def _cache_key(strategy, chunks):
    payload = json.dumps(
        {"strategy": strategy, "chunks": [c["text"] for c in chunks], "model": EMBED_MODEL},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def embed_chunks(client, strategy, chunks):
    key = _cache_key(strategy, chunks)
    cache = {}
    if CACHE_FILE.exists():
        cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    if cache.get(strategy, {}).get("key") == key:
        return np.array(cache[strategy]["embeddings"], dtype=np.float32)

    texts = [c["text"] for c in chunks]
    result = client.embed(texts, model=EMBED_MODEL, input_type="document")
    embeddings = np.array(result.embeddings, dtype=np.float32)

    cache[strategy] = {"key": key, "embeddings": embeddings.tolist()}
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache), encoding="utf-8")
    return embeddings


def cosine_similarity(query_vec, chunk_vecs):
    query_norm = query_vec / np.linalg.norm(query_vec)
    chunk_norms = chunk_vecs / np.linalg.norm(chunk_vecs, axis=1, keepdims=True)
    return chunk_norms @ query_norm


def format_handoff(query: str, strategy: str, results):
    lines = [f"=== RETRIEVED CONTEXT (strategy: {strategy}) ===\n"]
    for i, (chunk, score) in enumerate(results, 1):
        lines.append(f"[{i}] (source: {chunk['doc']}, similarity: {score:.3f}, len: {len(chunk['text'])} chars)")
        lines.append(chunk["text"])
        lines.append("")
    lines.append("=== QUERY ===")
    lines.append(query)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--strategy", choices=["fixed", "sentence", "recursive", "semantic"], default="recursive")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--dump-chunks", action="store_true", help="Print every chunk instead of retrieving")
    args = parser.parse_args()

    load_dotenv(dotenv_path=HERE.parent / ".env")
    client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])

    chunks = build_chunks(args.strategy, client)

    if args.dump_chunks:
        for i, c in enumerate(chunks):
            print(f"--- chunk {i} ({c['doc']}, {len(c['text'])} chars) ---")
            print(c["text"])
            print()
        print(f"Total chunks: {len(chunks)}", file=sys.stderr)
        return

    chunk_vecs = embed_chunks(client, args.strategy, chunks)
    query_result = client.embed([args.query], model=EMBED_MODEL, input_type="query")
    query_vec = np.array(query_result.embeddings[0], dtype=np.float32)

    scores = cosine_similarity(query_vec, chunk_vecs)
    top_indices = np.argsort(-scores)[:args.top_k]
    results = [(chunks[i], float(scores[i])) for i in top_indices]

    print(format_handoff(args.query, args.strategy, results))


if __name__ == "__main__":
    main()
