"""
Score retrieval methods from 01/02/04/05 against each other numerically,
using recall@k and MRR on the labeled, CATEGORIZED query set in
eval_set.py (easy / boundary / exact_token / compound -- see that file's
docstring for why each category exists), instead of reading retrieved
text and judging it by eye.

- Recall@k: for what fraction of queries does a relevant chunk appear
  ANYWHERE in the top k results?
- MRR (Mean Reciprocal Rank): average of 1/rank of the first relevant
  chunk found, across all queries (0 if never found in the ranked list
  considered). Rewards ranking the answer higher, not just including it.

Six methods are compared: all four chunking strategies from 02 (fixed,
sentence, recursive, semantic) scored with plain vector search, plus
hybrid (04: recursive chunking + BM25, RRF-fused) and reranked (05:
recursive chunking, vector top-10 reranked). This is deliberately more
methods and more (harder, categorized) queries than the original version
of this project -- a small, easy eval set makes every method look equally
good, which teaches nothing. See README for what the categories reveal
that a single flat query list didn't.

A chunk counts as "relevant" to a query if it contains that query's labeled
substring (see eval_set.py) after whitespace normalization -- this makes
relevance judgments comparable across chunking strategies with different
boundaries, since a fixed chunk index would not be.

Usage:
    python eval.py
"""

import hashlib
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import voyageai
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

from eval_set import EVAL_SET

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

HERE = Path(__file__).parent
DOCS_DIR = HERE / "data" / "docs"
CACHE_FILE = HERE / "data" / "cache" / "embeddings.json"
EMBED_MODEL = "voyage-3.5"
RERANK_MODEL = "rerank-2.5"
TOP_K = 3
CONSIDER_UP_TO = 10  # how deep into the ranking to look for MRR
CATEGORIES = ["easy", "boundary", "exact_token", "compound"]


# ---- chunking strategies (duplicated from 01/02 -- see those READMEs for why) ----

def chunk_fixed(text, doc_name, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            chunks.append({"doc": doc_name, "text": chunk})
        start += chunk_size - overlap
    return chunks


SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def split_sentences(text):
    text = text.strip()
    return [s.strip() for s in SENTENCE_RE.split(text) if s.strip()] if text else []


def chunk_sentence(text, doc_name, chunk_size=500):
    chunks, current = [], ""
    for sentence in split_sentences(text):
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) > chunk_size and current:
            chunks.append({"doc": doc_name, "text": current})
            current = sentence
        else:
            current = candidate
    if current:
        chunks.append({"doc": doc_name, "text": current})
    return chunks


def chunk_recursive(text, doc_name, chunk_size=500):
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


def chunk_semantic(text, doc_name, sentence_vecs, threshold_percentile=25):
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return [{"doc": doc_name, "text": text.strip()}] if text.strip() else []

    norm = sentence_vecs / np.linalg.norm(sentence_vecs, axis=1, keepdims=True)
    sims = np.sum(norm[:-1] * norm[1:], axis=1)
    threshold = np.percentile(sims, threshold_percentile)
    breakpoints = set(np.where(sims < threshold)[0] + 1)

    chunks, current = [], sentences[0]
    for i in range(1, len(sentences)):
        if i in breakpoints:
            chunks.append({"doc": doc_name, "text": current})
            current = sentences[i]
        else:
            current = f"{current} {sentences[i]}"
    chunks.append({"doc": doc_name, "text": current})
    return chunks


def load_and_chunk(chunk_fn):
    all_chunks = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        all_chunks.extend(chunk_fn(path.read_text(encoding="utf-8"), path.name))
    return all_chunks


def load_and_chunk_semantic(client):
    docs = sorted(DOCS_DIR.glob("*.md"))
    doc_texts = [(p.name, p.read_text(encoding="utf-8")) for p in docs]
    doc_sentences = [split_sentences(text) for _, text in doc_texts]
    flat_sentences = [s for sents in doc_sentences for s in sents]

    result = client.embed(flat_sentences, model=EMBED_MODEL, input_type="document")
    flat_vecs = np.array(result.embeddings, dtype=np.float32)

    all_chunks, offset = [], 0
    for (doc_name, text), sents in zip(doc_texts, doc_sentences):
        vecs = flat_vecs[offset:offset + len(sents)]
        offset += len(sents)
        all_chunks.extend(chunk_semantic(text, doc_name, vecs))
    return all_chunks


def tokenize(text):
    return re.findall(r"[a-z0-9][a-z0-9\-]*", text.lower())


def embed_texts(client, texts, cache_key):
    payload = json.dumps({"texts": texts, "model": EMBED_MODEL}, sort_keys=True)
    key = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    cache = {}
    if CACHE_FILE.exists():
        cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    if cache.get(cache_key, {}).get("key") == key:
        return np.array(cache[cache_key]["embeddings"], dtype=np.float32)

    result = client.embed(texts, model=EMBED_MODEL, input_type="document")
    embeddings = np.array(result.embeddings, dtype=np.float32)
    cache[cache_key] = {"key": key, "embeddings": embeddings.tolist()}
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache), encoding="utf-8")
    return embeddings


def is_relevant(chunk, item):
    normalized = re.sub(r"\s+", " ", chunk["text"])
    return chunk["doc"] == item["doc"] and item["substring"] in normalized


def vector_rank(query_vec, chunk_vecs):
    chunk_norm = chunk_vecs / np.linalg.norm(chunk_vecs, axis=1, keepdims=True)
    query_norm = query_vec / np.linalg.norm(query_vec)
    scores = chunk_norm @ query_norm
    order = np.argsort(-scores)
    return list(order), scores


def rrf_fuse(rankings, k=60):
    fused = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return [idx for idx, _ in sorted(fused.items(), key=lambda kv: -kv[1])]


def found_rank(chunks, ranking, item, consider_up_to=CONSIDER_UP_TO):
    for rank, idx in enumerate(ranking[:consider_up_to], 1):
        if is_relevant(chunks[idx], item):
            return rank
    return None


def score_method(name, chunks, rankings_per_query, per_query_log=None):
    """rankings_per_query: list of ranked chunk-index lists, one per eval query.
    Returns (overall_recall, overall_mrr, {category: (recall, mrr)})."""
    hits, ranks_by_cat, hits_by_cat = 0, defaultdict(list), defaultdict(int)

    for item, ranking in zip(EVAL_SET, rankings_per_query):
        rank = found_rank(chunks, ranking, item)
        if rank is not None and rank <= TOP_K:
            hits += 1
            hits_by_cat[item["category"]] += 1
        rr = 1.0 / rank if rank else 0.0
        ranks_by_cat[item["category"]].append(rr)
        if per_query_log is not None:
            per_query_log[item["query"]][name] = rank

    n = len(EVAL_SET)
    recall = hits / n
    mrr = sum(v for rrs in ranks_by_cat.values() for v in rrs) / n

    by_category = {}
    for cat in CATEGORIES:
        cat_items = [i for i in EVAL_SET if i["category"] == cat]
        cat_recall = hits_by_cat[cat] / len(cat_items) if cat_items else 0.0
        cat_mrr = sum(ranks_by_cat[cat]) / len(cat_items) if cat_items else 0.0
        by_category[cat] = (cat_recall, cat_mrr)

    print(f"{name:<16} recall@{TOP_K}: {recall:.2f}   MRR@{CONSIDER_UP_TO}: {mrr:.3f}   "
          + "  ".join(f"{cat}={by_category[cat][0]:.2f}/{by_category[cat][1]:.2f}" for cat in CATEGORIES))
    return recall, mrr, by_category


def main():
    load_dotenv(dotenv_path=HERE.parent / ".env")
    client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])

    # Setup does up to 6 embed calls (4 chunk strategies -- semantic needs its
    # own sentence-embedding pass first -- plus the query batch), which blows
    # through the free tier's 3 req/min limit if fired back to back. Pace
    # them; cached calls (unchanged chunk text) return instantly regardless.
    last_call = [0.0]

    def paced():
        elapsed = time.time() - last_call[0]
        if elapsed < 21:
            time.sleep(21 - elapsed)
        last_call[0] = time.time()

    fixed_chunks = load_and_chunk(chunk_fixed)
    sentence_chunks = load_and_chunk(chunk_sentence)
    recursive_chunks = load_and_chunk(chunk_recursive)

    paced()
    fixed_vecs = embed_texts(client, [c["text"] for c in fixed_chunks], "fixed")
    paced()
    sentence_vecs = embed_texts(client, [c["text"] for c in sentence_chunks], "sentence")
    paced()
    recursive_vecs = embed_texts(client, [c["text"] for c in recursive_chunks], "recursive")
    paced()
    semantic_chunks = load_and_chunk_semantic(client)
    paced()
    semantic_vecs = embed_texts(client, [c["text"] for c in semantic_chunks], "semantic")

    queries = [item["query"] for item in EVAL_SET]
    paced()
    query_result = client.embed(queries, model=EMBED_MODEL, input_type="query")
    query_vecs = np.array(query_result.embeddings, dtype=np.float32)

    bm25 = BM25Okapi([tokenize(c["text"]) for c in recursive_chunks])

    cat_counts = {cat: sum(1 for i in EVAL_SET if i["category"] == cat) for cat in CATEGORIES}
    cat_counts_str = ", ".join(f"{cat}={n}" for cat, n in cat_counts.items())
    print(f"Evaluating {len(EVAL_SET)} queries ({cat_counts_str}) "
          f"against {len(fixed_chunks)} fixed / {len(sentence_chunks)} sentence / "
          f"{len(recursive_chunks)} recursive / {len(semantic_chunks)} semantic chunks.\n")
    print(f"{'method':<16} {'recall@3':>9}  {'MRR@10':>7}   category recall/MRR "
          f"({'/'.join(CATEGORIES)})")
    print("-" * 110)

    per_query_log = defaultdict(dict)  # query -> {method: rank}

    naive_rankings = [vector_rank(qv, fixed_vecs)[0] for qv in query_vecs]
    score_method("naive (fixed)", fixed_chunks, naive_rankings, per_query_log)

    sentence_rankings = [vector_rank(qv, sentence_vecs)[0] for qv in query_vecs]
    score_method("sentence", sentence_chunks, sentence_rankings, per_query_log)

    recursive_rankings = [vector_rank(qv, recursive_vecs)[0] for qv in query_vecs]
    score_method("recursive", recursive_chunks, recursive_rankings, per_query_log)

    semantic_rankings = [vector_rank(qv, semantic_vecs)[0] for qv in query_vecs]
    score_method("semantic", semantic_chunks, semantic_rankings, per_query_log)

    hybrid_rankings = []
    for item, qv in zip(EVAL_SET, query_vecs):
        vec_ranking, _ = vector_rank(qv, recursive_vecs)
        bm25_scores = bm25.get_scores(tokenize(item["query"]))
        bm25_ranking = list(np.argsort(-bm25_scores))
        hybrid_rankings.append(rrf_fuse([vec_ranking, bm25_ranking]))
    score_method("hybrid", recursive_chunks, hybrid_rankings, per_query_log)

    print(f"\nReranking each of {len(EVAL_SET)} queries (paced, one Voyage rerank call "
          f"per query, ~{len(EVAL_SET)*21}s)...", file=sys.stderr)
    reranked_rankings = []
    for i, (item, qv) in enumerate(zip(EVAL_SET, query_vecs)):
        candidates, _ = vector_rank(qv, recursive_vecs)
        candidates = candidates[:10]
        texts = [recursive_chunks[c]["text"] for c in candidates]
        result = client.rerank(item["query"], texts, model=RERANK_MODEL, top_k=10)
        reranked_rankings.append([candidates[r.index] for r in result.results])
        if i < len(EVAL_SET) - 1:
            time.sleep(21)
    score_method("reranked", recursive_chunks, reranked_rankings, per_query_log)

    # -- rerank rank-movement detail: vector-only rank of the correct chunk vs. after reranking --
    print("\n\nReranking's effect, per query (recursive-chunking vector rank -> reranked rank):")
    print(f"{'category':<12} {'vector':>7} {'reranked':>9}  query")
    print("-" * 100)
    for item in EVAL_SET:
        before = per_query_log[item["query"]]["recursive"]
        after = per_query_log[item["query"]]["reranked"]
        before_s = str(before) if before else ">10"
        after_s = str(after) if after else ">10"
        moved = ""
        if before and after and after < before:
            moved = f"  (moved up {before - after})"
        elif before and after and after > before:
            moved = f"  (moved down {after - before})"
        print(f"{item['category']:<12} {before_s:>7} {after_s:>9}  {item['query'][:70]}{moved}")


if __name__ == "__main__":
    main()
