"""
Demonstrates hnswlib's save_index/load_index: build the HNSW graph once,
persist it to disk, and reload it on every later run instead of rebuilding
from scratch. Same idea as embeddings.json in 01/02/04 -- pay an expensive
step once, reuse the result on disk -- just for the GRAPH STRUCTURE this
time. (Rebuilding the graph from already-embedded vectors is the 19-second-
at-50k-vectors cost measured in benchmark_scaling.py; that's a separate,
even more avoidable cost than re-embedding text, which embeddings.json
already caches.)

Usage:
    python persistence_demo.py            # builds + saves on first run,
                                           # loads instead of rebuilding on
                                           # every run after
    python persistence_demo.py --rebuild  # delete the saved index and
                                           # force a fresh build
"""

import argparse
import sys
import time
from pathlib import Path

import hnswlib
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).parent
INDEX_FILE = HERE / "data" / "cache" / "hnsw_index.bin"
DIM = 1024           # matches voyage-3.5's embedding dimensionality
N = 50_000           # large enough that the build cost is worth persisting
TOP_K = 10


def build_corpus():
    """Deterministic clustered vectors (same shape as benchmark_scaling.py's
    realistic case) so build and load runs work with the identical corpus.
    """
    rng = np.random.default_rng(seed=42)
    centers = rng.normal(size=(50, DIM)).astype(np.float32)
    assignments = rng.integers(0, len(centers), size=N)
    noise = rng.normal(scale=0.15, size=(N, DIM)).astype(np.float32)
    return centers[assignments] + noise


def build_and_save():
    corpus = build_corpus()
    index = hnswlib.Index(space="cosine", dim=DIM)
    index.init_index(max_elements=N, ef_construction=200, M=16)

    start = time.perf_counter()
    index.add_items(corpus, np.arange(N))
    build_time = time.perf_counter() - start

    INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    index.save_index(str(INDEX_FILE))
    size_mb = INDEX_FILE.stat().st_size / (1024 * 1024)

    print(f"No saved index found -- built {N:,} vectors from scratch.")
    print(f"  build time: {build_time * 1000:.1f} ms")
    print(f"  saved to:   {INDEX_FILE.relative_to(HERE)}  ({size_mb:.1f} MB on disk)")
    return index


def load_saved():
    index = hnswlib.Index(space="cosine", dim=DIM)

    start = time.perf_counter()
    index.load_index(str(INDEX_FILE), max_elements=N)
    load_time = time.perf_counter() - start

    print(f"Found saved index at {INDEX_FILE.relative_to(HERE)} -- loading instead of rebuilding.")
    print(f"  load time:  {load_time * 1000:.1f} ms")
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="Delete the saved index and force a fresh build")
    args = parser.parse_args()

    if args.rebuild and INDEX_FILE.exists():
        INDEX_FILE.unlink()

    index = load_saved() if INDEX_FILE.exists() else build_and_save()

    index.set_ef(50)
    query = np.random.default_rng(seed=123).normal(size=(1, DIM)).astype(np.float32)

    start = time.perf_counter()
    labels, _ = index.knn_query(query, k=TOP_K)
    query_time = time.perf_counter() - start

    print(f"\nQuery against {index.get_current_count():,} vectors:")
    print(f"  query time: {query_time * 1000:.2f} ms")
    print(f"  top-{TOP_K} ids: {labels[0].tolist()}")


if __name__ == "__main__":
    main()
