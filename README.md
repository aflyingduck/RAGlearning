# RAG Learning

A hands-on curriculum for learning Retrieval-Augmented Generation (RAG) by
building progressively better versions of it. Each numbered folder is a
standalone mini-project: a specific RAG technique, built from scratch, with
a README explaining the concept and a concrete example of what it fixes
(and what it still doesn't).

## How each project works

Every project's script does **retrieval only** — chunk documents, embed
them, search, return the top results — and then stops. It never calls an
LLM to generate an answer. Instead it prints the retrieved context and your
query as a clearly delimited block that you paste into this chat (Claude
Code), and generation happens here, interactively.

This is intentional, not a missing feature. It keeps the two halves of RAG
visible and separately gradeable:

- **Retrieval** — chunking, embeddings, search. Runs as Python in each
  project folder. This is where almost all RAG quality problems actually
  live, and it's easy to hide that fact if an LLM call papers over bad
  retrieval with a fluent-sounding answer anyway.
- **Generation** — turning retrieved context into an answer. Done by Claude
  in this chat, using *only* what the retrieval step handed it.

## Stack

- **Python** (3.13, this environment) — all pipeline code
- **Voyage AI** (`voyage-3.5`) — embeddings. Anthropic recommends Voyage as
  its embeddings partner; Claude itself doesn't produce embeddings, since
  embedding and generation are different model architectures doing
  different jobs.
- **numpy** — vector math (cosine similarity), written out explicitly
  rather than hidden inside a vector-DB library, so the "search" step stays
  visible as what it actually is: a dot product.
- A few projects add one narrowly-scoped dependency each, installed only
  where needed rather than all upfront: `rank-bm25` (04, hybrid search),
  `hnswlib` (03, approximate nearest neighbor), `networkx` (10, graph
  traversal), `pillow` (11, generating/loading images). Reranking (05) and
  multimodal embedding (11) use Voyage's own `rerank` and
  `multimodal_embed` APIs — no local ML dependency needed.

## Setup

```
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt
```

Put your Voyage API key in `.env` at the repo root (already gitignored):
```
VOYAGE_API_KEY=pa-...
```
Get one free at [voyageai.com](https://www.voyageai.com/) — the free tier
(200M tokens) is plenty for this whole curriculum, though without a payment
method on file it's rate-limited to 3 requests/minute, which you'll notice
if you script rapid-fire test queries.

## Start here

Read [01-naive-rag](01-naive-rag/)'s README first and run its command —
it's the baseline every other project measures itself against. Then go in
order (02, 03, 04...): each later project assumes you've seen the ones
before it and builds on them directly (e.g. 05's reranking sits on top of
02's chunking; 06 scores 01/02/04/05 against each other).

## Curriculum

All 12 projects are built. Every technique below was verified against real
retrieval output before its README was written — not described in the
abstract — and several turned up genuine, sometimes counterintuitive
findings in the process (a benchmark bug in 03, a chunking-strategy
regression in 06, query transformation making things *worse* in 07). The
"key finding" column is the one-line version; each project's own README
has the real output behind it.

**Foundations**

| # | Project | Key finding |
|---|---------|-------------|
| 01 | [naive-rag](01-naive-rag/) | Fixed-size chunking cuts words in half at chunk boundaries — the baseline every later project improves on. |
| 02 | [chunking-strategies](02-chunking-strategies/) | Recursive (paragraph-first) chunking fixes the mid-word cut; semantic chunking groups by meaning but inherits every flaw of its sentence splitter. |
| 03 | [vector-stores-and-ann](03-vector-stores-and-ann/) | ANN (HNSW) only wins past ~10K+ vectors — at 20 chunks brute force is *faster*. A buggy recall benchmark taught more than a correct one would have. |
| 04 | [hybrid-search](04-hybrid-search/) | Vector search ranked the wrong error code's explanation above the right one; BM25 nailed the exact token instantly. RRF fusion combines both. |
| 05 | [reranking](05-reranking/) | A cross-encoder promoted the actually-correct chunk from rank #3 to #1 — reranking fixes ordering, but can't rescue what retrieval never found. |
| 06 | [evaluation](06-evaluation/) | Recall@3 saturated at 1.0 for every method (eval set was too easy) but MRR caught recursive chunking scoring *worse* than naive — reranking recovered it. |

**Advanced**

| # | Project | Key finding |
|---|---------|-------------|
| 07 | [query-transformation](07-query-transformation/) | Modern embeddings shrug off most paraphrasing. HyDE fixed a real compound-question failure; multi-query fusion made the same query *worse*. |
| 08 | [metadata-filtering](08-metadata-filtering/) | Two conflicting Hub-generation instructions scored within 0.008 of each other — filtering excludes the wrong one categorically, not probabilistically. |
| 09 | [agentic-rag](09-agentic-rag/) | Single-shot retrieval usually found the facts; what it can't do is the arithmetic, or notice an unverified constraint and check for it. |
| 10 | [graph-rag](10-graph-rag/) | Flat search surfaced a plausible-looking *wrong* number for a 3-hop question; graph traversal through opaque connector labels got the real answer. |
| 11 | [multimodal-rag](11-multimodal-rag/) | An image-only fact (a color legend) was retrieved correctly with a genuine, verified score gap — not a normalization artifact. |
| 12 | [production-concerns](12-production-concerns/) | Per-document hashing cut re-embedding to 0 API calls on a no-op run; a simulated ingestion bug was caught as a real score drop on exactly the affected queries. |

Curriculum order was foundations-first by design — each advanced project
leans on techniques (chunking, hybrid search, reranking, evaluation) built
earlier. Reusing that structure for a 13th technique, or going deeper on
any one project, is a reasonable way to keep extending this.
