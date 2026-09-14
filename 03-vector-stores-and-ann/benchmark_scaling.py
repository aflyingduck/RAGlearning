"""
Brute-force cosine search vs. approximate nearest neighbor (ANN), at scale.

01/02 searched 19-20 chunks with plain numpy -- fast enough that "search
algorithm" isn't a meaningful bottleneck yet. This script uses SYNTHETIC
random vectors (no embedding API calls needed) to find the corpus size
where that stops being true, and shows what an ANN index (hnswlib, using
the HNSW algorithm) trades away to stay fast past that point: exactness.

Usage:
    python benchmark_scaling.py
"""

import sys
import time

import hnswlib
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

DIM = 1024          # matches voyage-3.5's embedding dimensionality
NUM_QUERIES = 30
TOP_K = 10
SIZES = [1_000, 10_000, 50_000]


def brute_force_topk(corpus, queries, k):
    """Exact search: every query against every vector, via matrix multiply.
    This is exactly what 01/02's cosine_similarity() does, just batched
    over many queries at once instead of one at a time.
    """
    corpus_norm = corpus / np.linalg.norm(corpus, axis=1, keepdims=True)
    queries_norm = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    scores = queries_norm @ corpus_norm.T  # (num_queries, corpus_size)
    return np.argsort(-scores, axis=1)[:, :k]


def hnsw_topk(corpus, queries, k):
    """Approximate search via HNSW (Hierarchical Navigable Small World
    graph). Builds a graph index once, then each query walks the graph
    instead of comparing against every vector -- sublinear instead of
    linear in corpus size, at the cost of occasionally missing the true
    best match.
    """
    index = hnswlib.Index(space="cosine", dim=DIM)
    index.init_index(max_elements=len(corpus), ef_construction=200, M=16)
    build_start = time.perf_counter()
    index.add_items(corpus, np.arange(len(corpus)))
    build_time = time.perf_counter() - build_start

    index.set_ef(50)  # search-time recall/speed knob; higher = slower but more accurate
    query_start = time.perf_counter()
    labels, _ = index.knn_query(queries, k=k)
    query_time = time.perf_counter() - query_start

    return labels, build_time, query_time


def recall_at_k(exact, approx, k):
    """Fraction of the exact top-k that the approximate search also found,
    averaged over all queries. This is the metric ANN search sacrifices for
    speed -- 1.0 means the approximation found everything the exact search
    did.
    """
    hits = 0
    for e_row, a_row in zip(exact, approx):
        hits += len(set(e_row[:k]) & set(a_row[:k]))
    return hits / (len(exact) * k)


def generate_clustered(n, centers, rng, cluster_std=0.15):
    """Real embeddings aren't uniform random noise -- semantically similar
    text lands in genuine clusters, and a query embedding lands near the
    cluster(s) relevant to it. This generates synthetic vectors with that
    same clustered structure. Corpus and queries must share the same
    `centers` array, or the queries aren't actually related to the corpus
    -- that mistake is exactly what produced a misleadingly bad recall
    number during development of this script (see README).
    """
    assignments = rng.integers(0, len(centers), size=n)
    noise = rng.normal(scale=cluster_std, size=(n, centers.shape[1])).astype(np.float32)
    return centers[assignments] + noise


def ef_sweep(corpus, queries, exact, k, ef_values=(10, 20, 50, 100, 200, 400)):
    """`ef` is the search-time knob: how many candidate nodes HNSW keeps in
    play while walking the graph toward a query. Low ef = narrow, fast,
    prone to missing the true nearest neighbor if the graph walk goes down
    a slightly wrong path early. High ef = wider search, slower, closer to
    exact. Same index, same vectors, same queries -- only ef changes -- so
    this isolates exactly what that one knob buys and costs.
    """
    index = hnswlib.Index(space="cosine", dim=corpus.shape[1])
    index.init_index(max_elements=len(corpus), ef_construction=200, M=16)
    index.add_items(corpus, np.arange(len(corpus)))

    print(f"\n{'ef':>6} | {'query time':>11} | {'recall@10':>9}", flush=True)
    print("-" * 34, flush=True)
    for ef in ef_values:
        index.set_ef(max(ef, k))  # hnswlib requires ef >= k
        start = time.perf_counter()
        labels, _ = index.knn_query(queries, k=k)
        query_time = time.perf_counter() - start
        recall = recall_at_k(exact, labels, k)
        print(f"{ef:>6} | {query_time*1000:>8.2f} ms | {recall:>9.3f}", flush=True)


def main():
    rng = np.random.default_rng(seed=42)
    queries = rng.normal(size=(NUM_QUERIES, DIM)).astype(np.float32)

    print(f"{'N':>10} | {'brute-force':>12} | {'hnsw build':>11} | {'hnsw query':>11} | {'speedup':>8} | {'recall@10':>9}", flush=True)
    print("-" * 78, flush=True)

    for n in SIZES:
        corpus = rng.normal(size=(n, DIM)).astype(np.float32)

        bf_start = time.perf_counter()
        exact = brute_force_topk(corpus, queries, TOP_K)
        bf_time = time.perf_counter() - bf_start

        approx, build_time, query_time = hnsw_topk(corpus, queries, TOP_K)
        recall = recall_at_k(exact, approx, TOP_K)
        speedup = bf_time / query_time if query_time > 0 else float("inf")

        print(f"{n:>10,} | {bf_time*1000:>9.1f} ms | {build_time*1000:>8.1f} ms | "
              f"{query_time*1000:>8.1f} ms | {speedup:>7.1f}x | {recall:>9.3f}", flush=True)

    print("\nSame test, but with CLUSTERED vectors sharing centers with their queries", flush=True)
    print("(like real embeddings, where a query really does land near relevant chunks):", flush=True)
    n = 10_000
    cluster_centers = rng.normal(size=(50, DIM)).astype(np.float32)
    corpus = generate_clustered(n, cluster_centers, rng)
    cluster_queries = generate_clustered(NUM_QUERIES, cluster_centers, rng)

    exact = brute_force_topk(corpus, cluster_queries, TOP_K)
    approx, build_time, query_time = hnsw_topk(corpus, cluster_queries, TOP_K)
    recall = recall_at_k(exact, approx, TOP_K)
    print(f"{n:>10,} | {'(see above)':>12} | {build_time*1000:>8.1f} ms | "
          f"{query_time*1000:>8.1f} ms | {'':>8} | {recall:>9.3f}", flush=True)

    print("\nThe run above used a fixed ef=50. Sweeping ef on the SAME clustered")
    print("index/queries shows the actual speed-vs-recall dial ANN search gives you:", flush=True)
    ef_sweep(corpus, cluster_queries, exact, TOP_K)


if __name__ == "__main__":
    main()
