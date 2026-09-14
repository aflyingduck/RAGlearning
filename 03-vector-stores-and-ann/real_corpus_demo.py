"""
The same brute-force-vs-hnsw comparison as benchmark_scaling.py, but on the
real 20-chunk Nimbus Hub corpus instead of synthetic vectors -- to show
that at THIS scale, the choice of search algorithm is invisible. The
benchmark's speedup and recall numbers only start to matter once your
corpus looks more like 100k+ chunks than 20.

Usage:
    python real_corpus_demo.py "your question"
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import hnswlib
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


def chunk_recursive(text: str, doc_name: str, chunk_size=CHUNK_SIZE):
    """Same paragraph-first splitter introduced in 02-chunking-strategies --
    duplicated here (rather than imported across project folders) so this
    project stays runnable on its own.
    """
    import re
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


def brute_force_search(query_vec, chunk_vecs, k):
    chunk_norm = chunk_vecs / np.linalg.norm(chunk_vecs, axis=1, keepdims=True)
    query_norm = query_vec / np.linalg.norm(query_vec)
    scores = chunk_norm @ query_norm
    top = np.argsort(-scores)[:k]
    return top, scores[top]


def hnsw_search(query_vec, chunk_vecs, k):
    index = hnswlib.Index(space="cosine", dim=chunk_vecs.shape[1])
    index.init_index(max_elements=len(chunk_vecs), ef_construction=200, M=16)
    index.add_items(chunk_vecs, np.arange(len(chunk_vecs)))
    index.set_ef(50)
    labels, distances = index.knn_query(query_vec.reshape(1, -1), k=k)
    return labels[0], 1 - distances[0]  # hnswlib cosine space returns distance = 1 - similarity


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
