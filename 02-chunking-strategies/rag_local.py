"""
Same chunking-strategy comparison as rag.py, but embeddings come from a
local model (BAAI/bge-small-en-v1.5, run on-device via fastembed) instead
of Voyage's API. No API key, no network call at query time.

Run the same query/strategy through both to compare:
    python rag.py "query" --strategy semantic
    python rag_local.py "query" --strategy semantic
"""

import argparse
import hashlib
import json
import sys

import numpy as np
from fastembed import TextEmbedding

from chunkers import NON_SEMANTIC_STRATEGIES, chunk_semantic, split_sentences
from rag import DOCS_DIR, CHUNK_SIZE, HERE, cosine_similarity, format_handoff

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

CACHE_FILE_LOCAL = HERE / "data" / "cache" / "embeddings_local.json"
EMBED_MODEL_LOCAL = "BAAI/bge-small-en-v1.5"


def build_chunks_local(strategy: str, model):
    """Same dispatch as rag.py's build_chunks, but semantic's sentence
    embeddings come from the local model's passage_embed instead of Voyage.
    """
    all_chunks = []
    docs = sorted(DOCS_DIR.glob("*.md"))

    if strategy == "semantic":
        doc_texts = [(path.name, path.read_text(encoding="utf-8")) for path in docs]
        doc_sentences = [split_sentences(text) for _, text in doc_texts]
        flat_sentences = [s for sents in doc_sentences for s in sents]

        flat_vecs = np.array(list(model.passage_embed(flat_sentences)), dtype=np.float32)

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
        {"strategy": strategy, "chunks": [c["text"] for c in chunks], "model": EMBED_MODEL_LOCAL},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def embed_chunks_local(model, strategy, chunks):
    key = _cache_key(strategy, chunks)
    cache = {}
    if CACHE_FILE_LOCAL.exists():
        cache = json.loads(CACHE_FILE_LOCAL.read_text(encoding="utf-8"))
    if cache.get(strategy, {}).get("key") == key:
        return np.array(cache[strategy]["embeddings"], dtype=np.float32)

    texts = [c["text"] for c in chunks]
    embeddings = np.array(list(model.passage_embed(texts)), dtype=np.float32)

    cache[strategy] = {"key": key, "embeddings": embeddings.tolist()}
    CACHE_FILE_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE_LOCAL.write_text(json.dumps(cache), encoding="utf-8")
    return embeddings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--strategy", choices=["fixed", "sentence", "recursive", "semantic"], default="recursive")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--dump-chunks", action="store_true", help="Print every chunk instead of retrieving")
    args = parser.parse_args()

    model = TextEmbedding(model_name=EMBED_MODEL_LOCAL)

    chunks = build_chunks_local(args.strategy, model)

    if args.dump_chunks:
        for i, c in enumerate(chunks):
            print(f"--- chunk {i} ({c['doc']}, {len(c['text'])} chars) ---")
            print(c["text"])
            print()
        print(f"Total chunks: {len(chunks)}", file=sys.stderr)
        return

    chunk_vecs = embed_chunks_local(model, args.strategy, chunks)
    query_vec = np.array(list(model.query_embed([args.query]))[0], dtype=np.float32)

    scores = cosine_similarity(query_vec, chunk_vecs)
    top_indices = np.argsort(-scores)[:args.top_k]
    results = [(chunks[i], float(scores[i])) for i in top_indices]

    print(format_handoff(args.query, args.strategy, results))


if __name__ == "__main__":
    main()
