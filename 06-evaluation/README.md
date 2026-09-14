# 06 — Evaluation

Every project so far judged retrieval quality by reading printed chunks and
deciding "does this look right." That doesn't scale, doesn't catch
regressions, and doesn't let you compare two methods objectively. This
project replaces eyeballing with two standard retrieval metrics, computed
over a labeled query set, and scores **six methods** — all four chunking
strategies from 02 (fixed, sentence, recursive, semantic) plus 04's hybrid
and 05's reranking — against each other.

```
cd 06-evaluation
../.venv/Scripts/python.exe eval.py
```

An earlier version of this project used 10 easy, hand-picked queries and
only compared 2 methods (fixed vs. recursive). Recall@3 saturated at 1.00
for everything — every method "passed," which taught nothing about *where*
each one actually struggles. This version fixes that by deliberately
building queries designed to separate the methods, not just confirm they
all work.

## The metrics

- **Recall@k** — for what fraction of queries does a relevant chunk appear
  *anywhere* in the top k results? Answers "did retrieval find it at all."
- **MRR (Mean Reciprocal Rank)** — average of `1/rank` of the first
  relevant chunk, across all queries (0 if it's never found within the
  window considered). Recall@k is blind to whether the answer was rank 1
  or rank k; MRR rewards ranking it higher.

## The labeled set (`eval_set.py`) — now categorized, and why

18 queries in 4 categories, each chosen to stress a *specific* mechanism
instead of testing "does retrieval work in general":

- **`easy`** (6 queries) — a clean, single fact, unique vocabulary. The
  floor every method should clear. Not where the interesting differences
  are — included so a method's failure elsewhere isn't confused with a
  method that's broken outright.
- **`boundary`** (4 queries) — the answer text sits exactly across a real
  `chunk_fixed()` cut point, *verified against actual chunking output*
  (not guessed — see below), so fixed chunking's specific known failure
  mode (01's mid-word cut) gets tested directly rather than hoped-for.
- **`exact_token`** (4 queries) — hinges on an opaque token (an error
  code, a protocol version like "TLS 1.3") with little semantic content of
  its own — 04-hybrid-search's exact failure mode for vector-only search,
  now tested with 4 queries instead of 1.
- **`compound`** (4 queries) — the answer sentence sits inside a paragraph
  that also discusses a related-but-different aspect of the same topic
  (reused patterns from 05 and 07, where this specific shape caused real
  ranking problems), so a single embedding has to represent one specific
  fact while sitting next to more generic neighboring text.

Substring containment — not chunk index — is still the relevance judgment,
since chunk boundaries differ across all four chunking strategies being
compared; a "correct chunk index" for one strategy is meaningless for
another.

The corpus also gained an "Error codes" section in `troubleshooting.md`
(same three fictional codes as 04) specifically to give the `exact_token`
category something real to test against — this project's corpus copy is
independent from 01-05's, so this doesn't touch or invalidate any other
project's numbers.

## Real results

```
Evaluating 18 queries (easy=6, boundary=4, exact_token=4, compound=4)
against 20 fixed / 20 sentence / 22 recursive / 22 semantic chunks.

method            recall@3   MRR@10   easy        boundary    exact_token compound
naive (fixed)     1.00       0.972    1.00/1.00   1.00/0.88   1.00/1.00   1.00/1.00
sentence          1.00       0.972    1.00/1.00   1.00/1.00   1.00/0.88   1.00/1.00
recursive         1.00       0.880    1.00/1.00   1.00/0.83   1.00/0.88   1.00/0.75
semantic          0.94       0.903    1.00/1.00   1.00/0.88   1.00/0.88   0.75/0.81
hybrid            0.94       0.931    1.00/1.00   1.00/0.88   1.00/1.00   0.75/0.81
reranked          1.00       1.000    1.00/1.00   1.00/1.00   1.00/1.00   1.00/1.00
```
(cells are `recall@3/MRR@10` per category)

This is a genuinely different picture than the old 10-query version — every
category except `easy` now separates the methods.

### `boundary`: recall survives, rank doesn't

Every method still hits recall@3 = 1.00 on boundary queries — 02's README
already found that 50-char overlap rescues *recall* from fixed chunking's
mid-word cuts. But **MRR tells a different story**: `naive (fixed)` drops
to 0.88 and `recursive` drops further to 0.83, while `sentence` — which
never splits mid-sentence at all — hits a perfect 1.00. The chunk still
technically contains the answer, but the mid-word garbling and pasted-
together neighboring text push it down a rank or two. This is the
honest, precise version of what 01/02 could only show qualitatively: fixed
and recursive chunking's boundary damage costs you *rank*, not recall,
on this corpus (a real production corpus without 01's generous overlap
parameter could easily lose recall entirely).

### `exact_token`: hybrid's whole reason to exist, now with 4 data points

`hybrid` is the only method to hit a perfect 1.00/1.00 on `exact_token` —
every pure-vector method (fixed, sentence, recursive, semantic) sits at
0.88-1.00 MRR, never quite as clean. This is 04's single-query
`"ERR-1090"` finding, now confirmed across 4 independent exact-token
queries instead of one example that could have been a fluke.

### `compound`: the category that actually separates everything

This is where the real variance lives:
- `recursive` keeps recall@3 = 1.00 but its MRR craters to **0.75** — the
  worst score any vector-only method posts in any category. Its
  paragraph-merged chunks (02's "keeps related sentences together"
  design) dilute a specific answer with adjacent generic content often
  enough to matter.
- `semantic` and `hybrid` both **drop to 0.75 recall** on this category —
  each missed one compound query's answer chunk *entirely* out of the top
  3, not just ranked it lower. Recall regressions are worse than MRR
  regressions: the answer isn't just buried, it's gone from what a
  downstream LLM would even see.
- `reranked` posts a perfect 1.00/1.00 — recovering every point recursive
  chunking gave up, on the exact category where it gave up the most.

### A real surprise this run turned up: hybrid regressed on `compound`

`hybrid`'s vector component uses the same `recursive` chunking, and
`recursive` alone gets **1.00 recall** on `compound` (all 4 answers
present in top-3, just ranked poorly — MRR 0.75). After RRF-fusing in
BM25, `hybrid`'s compound recall **drops to 0.75** — fusion didn't just
fail to help here, it pushed a chunk that vector search alone had
correctly kept in the top 3 *out* of it. This is the same mechanism 07
found with multi-query fusion: RRF combines two *ranked lists*, and if
BM25 has no real opinion on a paraphrased, compound question (no strong
keyword overlap to anchor on), blending in its noisy ranking can dilute a
vector ranker that was already doing fine. Hybrid search's win on
`exact_token` and its loss on `compound` are the same underlying
mechanism from opposite sides — BM25 helps when it has a confident,
correct opinion, and can hurt when it doesn't but still gets a vote.

## Reranking's effect, query by query

```
Reranking's effect, per query (recursive-chunking vector rank -> reranked rank):
category      vector  reranked  query
boundary           3         1  Is my voice data ever shared with other Solstice Labs customers?  (moved up 2)
exact_token         2         1  ERR-1090  (moved up 1)
compound            2         1  Do I need a subscription for local automations to work?  (moved up 1)
compound            2         1  I have a huge house and one of these little boxes probably can't cover...  (moved up 1)
```
(the other 14 of 18 queries were already at rank 1 under plain recursive
vector search and stayed there — reranking doesn't reorder what's already
correct)

Every single movement happened in `boundary`, `exact_token`, or
`compound` — **never once in `easy`.** That's not a coincidence; it's the
category design working as intended, and it's a stronger, more specific
claim than "reranking helps": reranking's entire measured benefit here
occurred exactly where 05's README predicted it would — cases where a
generically-topical chunk outranks the chunk that actually answers the
question. The first row is the exact query 05 used as its worked example
(rank 3 → rank 1); this harness now confirms that wasn't a cherry-picked
example — it's one of four real, reproducible corrections.

## Extending this

This harness only measures retrieval. It doesn't score the generated
answer (faithfulness, whether it stuck to the retrieved context, whether
it hallucinated) — that would need grading the text Claude produces in
this chat against the retrieved context, which is a manual/interactive
exercise given this curriculum's retrieval-only-script design (see the
top-level README). It also doesn't test `08`'s metadata filtering, `10`'s
graph traversal, or `11`'s multimodal retrieval — those techniques solve
problems (cross-document contamination, multi-hop relationships, non-text
content) that a flat chunk-substring eval set structurally can't
represent; each of their own READMEs verifies them with problem-specific
methodology instead. If you want to push this further yourself: the
`compound` category was the most revealing one built here — writing 4-6
more queries in that shape (or a `multi_hop` category modeled on 10's
graph-rag scenario) is the highest-leverage next addition.
