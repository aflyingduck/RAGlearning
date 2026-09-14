# 04 — Hybrid Search (BM25 + Vector Search)

Every project so far has searched by meaning alone (cosine similarity over
embeddings). This project adds BM25 — the keyword-ranking algorithm behind
decades of classic search engines — and combines it with vector search via
Reciprocal Rank Fusion (RRF), because the two methods fail in different,
complementary ways.

```
cd 04-hybrid-search
../.venv/Scripts/python.exe rag.py "your question" --mode vector|bm25|hybrid
```

## Why vector search alone isn't enough

The corpus's `troubleshooting.md` got a new "Error codes" section for this
project, listing three fictional firmware error codes: `ERR-4471`,
`ERR-2208`, `ERR-1090`. Ask vector search directly for one of them:

```
../.venv/Scripts/python.exe rag.py "ERR-1090" --mode vector --top-k 3
```

Real output:
```
[1] (source: troubleshooting.md, score: 0.4024)
**Error codes.** ... - `ERR-4471` — firmware signature verification failed...

[2] (source: troubleshooting.md, score: 0.4011)
... - `ERR-1090` — Hub Mesh link rejected because the two hubs are running
  different firmware versions...
```

**The chunk that actually explains `ERR-1090` ranks #2, behind the chunk
about a completely different error code** (`ERR-4471`), by a margin of
0.0013 — a coin flip. This is a real, reproducible failure, not a
contrived one. Embedding models represent meaning, and "ERR-4471" and
"ERR-1090" mean almost the same thing to an embedding: both are
"an error code in a troubleshooting doc." The specific digits — the part
that actually answers the question — carry very little semantic weight.
Vector search is bad at exactly the kind of exact-identifier matching that
error codes, SKUs, product model numbers, and ticket IDs need.

## BM25 gets it right, decisively

```
../.venv/Scripts/python.exe rag.py "ERR-1090" --mode bm25 --top-k 3
```
```
[1] (source: troubleshooting.md, score: 3.1464)
... - `ERR-1090` — Hub Mesh link rejected...

[2] (source: installation-guide.md, score: 0.0000)
[3] (source: installation-guide.md, score: 0.0000)
```

BM25 scores by term overlap (roughly: term frequency in the chunk, inverse
frequency across the corpus, normalized for chunk length). `"ERR-1090"` is
a token that appears in exactly one chunk in the whole corpus, so that
chunk wins overwhelmingly (3.15 vs. 0.0 for everything else) — there's no
ambiguity once you're matching exact tokens instead of nearby meanings.
This is also BM25's own weakness turned inside out: it has *no* opinion
about a query like "how do I stop the hub losing connection," where no
single keyword pins down the right chunk but the meaning clearly does
(vector search's strength — see 01/02).

## Combining them: Reciprocal Rank Fusion

`--mode hybrid` runs both searches and merges the two RANKED LISTS (not the
raw scores — BM25 scores and cosine similarities aren't on comparable
scales, so averaging them directly would be meaningless). RRF instead
gives each chunk `1 / (k + rank)` points per ranker it appears in and sums
across rankers, so a chunk that ranks well in *either* list scores well
overall, and a chunk both rankers agree on scores best of all:

```
../.venv/Scripts/python.exe rag.py "ERR-1090" --mode hybrid --top-k 3
```
```
[1] (source: troubleshooting.md, score: 0.0325)
... - `ERR-1090` — Hub Mesh link rejected because the two hubs are running
  different firmware versions...

[2] (source: installation-guide.md, score: 0.0308)
[3] (source: installation-guide.md, score: 0.0298)
```

Fixed: the correct chunk is back at #1. BM25's decisive, unambiguous
top pick (score 3.15 vs. 0.0 for everything else) dominates the fused
ranking even though vector search alone had it ranked #2 by a hair.

## When to reach for this

Hybrid search costs you: a second index to build and keep in sync
(`rank_bm25.BM25Okapi` here — cheap at this corpus size, but it's still a
second data structure), and a fusion step to tune. It's worth it whenever
your corpus mixes prose (where meaning matters) with exact identifiers
(where it doesn't) — which describes most real internal-docs, support, or
product-knowledge corpora. If your corpus and queries are pure prose with
no codes, SKUs, or jargon acronyms, vector search alone (01/02/03) may
genuinely be enough, and adding BM25 mainly adds maintenance surface for
little gain.
