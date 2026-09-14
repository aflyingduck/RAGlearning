# 03 — Vector Stores and Approximate Nearest Neighbor (ANN)

01/02 searched 19-20 chunks with a plain numpy matrix multiply and it was
instant. This project answers two questions: at what scale does that stop
being true, and what do you actually give up when you switch to an
approximate index to fix it?

Three scripts:
- **`benchmark_scaling.py`** — synthetic vectors, no embedding API calls, so
  it's free to run and safe to scale up. Compares brute-force numpy search
  against `hnswlib` (HNSW — Hierarchical Navigable Small World graphs) as
  corpus size grows, including a sweep of the `ef` search-width parameter.
- **`real_corpus_demo.py "query"`** — the same comparison on the real
  20-chunk Nimbus Hub corpus, needing 2 Voyage API calls (chunk + query
  embedding, cached after the first run).
- **`persistence_demo.py`** — builds a 50k-vector HNSW index, saves it to
  disk, and on every later run loads it instead of rebuilding. Shows where
  the built graph actually lives and what `save_index`/`load_index` buys
  you.

## What brute force and HNSW actually do

**Brute force** is `cosine_similarity()` from 01/02, just batched: normalize
every vector, matrix-multiply the query against *all* of them, sort. To
answer one query it computes N distances for a corpus of size N. No
cleverness, no setup cost, and it's always exactly correct — the top-k it
returns really are the top-k. Its only weakness is that cost scales
linearly: 10x the corpus, 10x the work, forever.

**HNSW** (Hierarchical Navigable Small World) avoids touching every vector
by pre-organizing them into a graph at index-build time, then *walking*
that graph at query time instead of scanning it:

- Picture a few stacked layers. The top layer has only a handful of nodes
  connected by long-range links — think highway system, a few hops get you
  roughly anywhere. Each layer down has more nodes and shorter, denser
  links — city streets, then footpaths. Every vector lives in the bottom
  layer; only a random subset also gets placed in the layers above it.
- A query starts at one entry point in the top (sparsest) layer and greedily
  hops to whichever neighbor is closer to the query than where it currently
  is. When no neighbor improves things, it drops down a layer and keeps
  going, now taking smaller, more precise steps. By the bottom layer it's
  already in roughly the right neighborhood, so it only has to examine a
  small local patch of the graph — not the whole corpus.
- That's *why* it's sublinear: a well-built graph gets you from "anywhere"
  to "roughly the right neighborhood" in a handful of big jumps (`O(log N)`-
  ish), rather than one comparison per vector.
- It's *approximate* because greedy hill-climbing on a graph can get stuck
  in a locally-good spot that isn't the actual global best — like always
  taking the road that looks fastest right now and occasionally missing a
  shortcut you couldn't see from where you were standing.

Three parameters in `hnsw_topk()` and `real_corpus_demo.py`'s `hnsw_search()`
control this trade-off directly:

| Param | Set when | What it controls |
|---|---|---|
| `M` | index build | Max graph connections per node. Higher = denser graph, better recall, more memory, slower build. |
| `ef_construction` | index build | How exhaustively the graph is searched *while inserting each vector*. Higher = better-quality graph, much slower build (this is why 50,000 vectors took 19 seconds below). |
| `ef` | every query | How wide a candidate set the graph walk keeps at query time. Higher = closer to exact, slower per query. This is the only one of the three you can change *after* the index is already built. |

## Brute force vs. HNSW as corpus size grows

```
../.venv/Scripts/python.exe benchmark_scaling.py
```
Real output, on this machine, with 1024-dim vectors (matching `voyage-3.5`)
and top-10 search:

```
         N |  brute-force |  hnsw build |  hnsw query |  speedup | recall@10
------------------------------------------------------------------------------
     1,000 |       6.5 ms |     25.3 ms |      3.1 ms |     2.1x |     0.840
    10,000 |      36.3 ms |   1106.8 ms |     13.9 ms |     2.6x |     0.243
    50,000 |     161.9 ms |  18948.5 ms |     23.8 ms |     6.8x |     0.097
```

Two real, measured trends:

- **Query speedup grows with N.** Brute force scales roughly linearly with
  corpus size (6.5 → 36 → 162 ms, roughly ×6 per ×10 growth in N — a matrix
  multiply's cost is proportional to corpus size). HNSW query time grows
  much more slowly (3 → 14 → 24 ms) because it doesn't compare against
  every vector — it walks a graph toward the answer, sublinear in N. That
  gap is *why* ANN indexes exist.
- **Index build is not free**, and it gets expensive fast: 25 ms at 1,000
  vectors, 19 *seconds* at 50,000. Every vector added to an HNSW index pays
  a one-time graph-construction cost. This matters directly for project 12
  (production-concerns): re-embedding and re-indexing a large corpus from
  scratch on every doc change is not something you want to do casually.

## The recall numbers above are a trap — and that's the actual lesson

Recall@10 *drops* as N grows (0.84 → 0.24 → 0.10), which looks like HNSW
getting worse at scale. It isn't. The vectors in that benchmark are pure
random Gaussian noise with no structure at all. In high-dimensional space
(1024-D here), random vectors are all roughly equidistant from each other —
cosine similarities cluster tightly around 0 regardless of which pair you
pick. That means "the true top-10 nearest neighbors" of a random query are
barely different from the 11th through 1000th closest points; the ranking
has no real signal for any algorithm, exact or approximate, to lock onto.
Tiny floating-point and graph-traversal differences reshuffle a ranking
that was never meaningful to begin with. This isn't an ANN weakness — ask
brute force to defend its own top-10 against a slightly different random
seed and it wouldn't do any better.

Real embeddings aren't like this — semantically similar text really does
cluster together in embedding space. The script's last section generates
synthetic vectors with genuine cluster structure (50 clusters, corpus and
queries drawn from the *same* centers) to check this directly:

```
Same test, but with CLUSTERED vectors sharing centers with their queries
(like real embeddings, where a query really does land near relevant chunks):
    10,000 |  (see above) |    267.8 ms |      4.2 ms |          |     0.997
```

99.7% recall, and both the index build (268 ms vs. 1.1 seconds for
unstructured data at the same N) and query time (4.2 ms vs. 14 ms) got
*faster* — real structure makes both search and indexing easier. This
version of the test is a fair proxy for real embeddings; the pure-noise
version above isn't. (This also cost some debugging time to get right —
an earlier draft of this script generated the query vectors' clusters from
a freshly-sampled random center set instead of the corpus's actual centers,
so the "clustered" queries weren't really related to the "clustered"
corpus at all, and recall collapsed to ~0.48 for the same reason as the
pure-noise case. If your own recall benchmark ever looks inexplicably bad,
check that your test queries are actually related to your test corpus
before blaming the index.)

### The `ef` knob, made concrete

Everything above used one fixed `ef=50`. Same clustered index, same
queries, only `ef` changing, sweeping it from 10 to 400:

```
    ef |  query time | recall@10
----------------------------------
    10 |     2.34 ms |     0.890
    20 |     2.03 ms |     0.963
    50 |     2.51 ms |     0.997
   100 |     3.09 ms |     1.000
   200 |     4.56 ms |     1.000
   400 |     9.55 ms |     1.000
```

This is the actual trade HNSW is offering you, isolated from every other
variable: at `ef=10` you miss roughly 1 in 10 of the true top-10 results
but every query costs ~2ms; by `ef=100` you stop missing anything, at
roughly 1.5x the query cost; past that, `ef=400` buys you nothing more
correctness-wise here and just burns time widening a search that's already
found everything. In production you'd pick the smallest `ef` that clears
your recall bar for *your* data and query patterns — there's no universally
correct value, and it's the one HNSW parameter you can tune after the
index is already built, without re-indexing anything.

## Where does the built graph actually live?

Every script above rebuilds the HNSW graph from scratch, every run, in
process memory — `hnsw_topk()` and `hnsw_search()` both call `init_index()`
+ `add_items()` fresh each time, then the process exits and the graph is
gone. That means the "build time" numbers throughout this README are being
paid on *every single query* in these demos, which overstates HNSW's
per-query cost and understates its real benefit: a system that serves many
queries builds the graph once and keeps reusing it.

`persistence_demo.py` shows the actual fix, using `hnswlib`'s built-in
`save_index()` / `load_index()` — the same "pay an expensive step once,
cache the result on disk" idea as `embeddings.json`, just applied to the
*graph structure* instead of the vectors:

```
../.venv/Scripts/python.exe persistence_demo.py --rebuild   # first run
No saved index found -- built 50,000 vectors from scratch.
  build time: 3259.1 ms
  saved to:   data\cache\hnsw_index.bin  (202.4 MB on disk)

../.venv/Scripts/python.exe persistence_demo.py             # every run after
Found saved index at data\cache\hnsw_index.bin -- loading instead of rebuilding.
  load time:  214.2 ms
```

~15x faster to load than to rebuild, and the query returns the *identical*
top-10 ids either way — reloading really does restore the same graph, not
a fresh approximation of it. (Note this run used clustered, not pure-noise,
vectors, so 3.3s isn't directly comparable to the 19s pure-noise build
earlier in this README — structured data builds faster too, per the
recall-trap section above. The build-vs-load *ratio* is the point here,
not the absolute number.)

That 202MB is also worth noticing: persisting the graph isn't free either —
it's roughly the size of the raw vectors themselves plus the graph's
connection data. `.gitignore` already excludes `**/data/cache/` in this
repo, so this file never gets committed.

So, concretely: **at rest**, the built graph is a file on disk (once you
call `save_index`); **at query time**, it's sitting in the RAM of whatever
process loaded it — built or loaded once, then reused across many queries.
That's also exactly what a production vector database (Pinecone, Qdrant,
Weaviate, or a self-hosted FAISS/hnswlib service) is, structurally: a
long-running server process holding a loaded index in memory, backed by a
persisted copy on disk for restarts, instead of a script that rebuilds
per invocation.

## Where this file lives in a real corporate system

Our demo's `data/cache/hnsw_index.bin` sits on one developer's laptop disk.
A real system never leaves it there. Roughly in order of how common each
option actually is:

**1. A managed vector database** (Pinecone, Weaviate Cloud, Qdrant Cloud,
Databricks Vector Search, Azure AI Search, AWS OpenSearch, ...). You never
touch an index file at all — you call an API to upsert vectors and query.
The provider handles the HNSW-or-similar index internally: where it's
stored, sharding, replication, backups. This is the default for most
corporate RAG systems specifically so nobody has to reason about the rest
of this section.

**2. A self-hosted vector database** (Qdrant/Weaviate/Milvus running in
your own Kubernetes cluster, say). The file lives somewhere concrete now:
- **On disk**: a *persistent volume* attached to the pod/VM (an AWS EBS
  volume, GCP Persistent Disk, Kubernetes `PersistentVolumeClaim`) — never
  the container's ephemeral local storage, which gets wiped on every
  restart/redeploy/autoscale event and would force a full rebuild each time.
- **In memory**: the database process loads that on-disk index into RAM at
  startup (exactly our `load_index()` demo above) and serves queries out
  of RAM from then on. Disk is the durable copy; RAM is what actually
  answers queries fast.

**3. Fully self-rolled** (raw `hnswlib`/FAISS, closer to what this project
does). The common pattern:
- **Source of truth**: the index file lives in *object storage* — S3, GCS,
  Azure Blob — versioned, so a bad rebuild can be rolled back and no single
  machine dying loses it.
- **At startup**, each serving instance downloads the current version from
  object storage and calls `load_index()` once.
- **On rebuild** (new docs, a schedule, or a triggered pipeline run), a
  separate batch job builds the new index, uploads it as a new version, and
  serving instances reload it on their next restart or a hot-reload signal.

**What's different because it's corporate, not a demo:**
- **Multiple replicas.** For uptime and throughput you run several serving
  instances, each holding its *own full copy* of the index in RAM — that's
  `load_index()` happening once per replica, not one graph shared across
  machines.
- **Updates aren't free.** `hnswlib` supports incremental `add_items()` /
  `mark_deleted()` without a full rebuild, but many teams still do periodic
  full rebuilds anyway because incremental updates degrade graph quality
  over time — an open tradeoff, not a solved problem.
- **Data residency / compliance.** The index file encodes information
  derived from the source documents — embeddings are a lossy but real
  representation of that content. Which region it's stored in, whether
  it's encrypted at rest, and who has read access falls under the same
  compliance rules as the source documents, and is often a bigger
  constraint on "where" than any technical concern above.

## At 20 chunks, none of this matters yet

```
../.venv/Scripts/python.exe real_corpus_demo.py "How long is the warranty on the Hub 3?" --top-k 3
```
```
Corpus size: 20 chunks

Brute-force: 0.443 ms
  [17] 0.711  warranty-and-returns.md: ...
  [18] 0.646  warranty-and-returns.md: ...
  [8]  0.643  product-overview.md: ...

HNSW (approximate): 1.429 ms
  [17] 0.711  ...
  [18] 0.646  ...
  [8]  0.643  ...

Same top-3 chunks: True
```

Identical results — and brute force is *faster*, because HNSW has to build
its graph index before it can search, and that setup cost dwarfs a 20×1024
matmul. This is the real takeaway: don't reach for a vector database or ANN
library until brute-force numpy search is measurably too slow for your
corpus. Every project in this curriculum so far has stayed well under that
threshold on purpose. The `benchmark_scaling.py` numbers above are what
"too slow" starts to look like — tens of thousands of chunks, not tens.
