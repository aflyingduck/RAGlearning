# 05 — Reranking

Vector search (01-04) scores every chunk against a query independently:
embed the chunk once, embed the query once, compare. A reranker scores each
(query, chunk) PAIR jointly — it's a cross-encoder, reading both texts
together, so it can weigh how specifically a chunk answers *this* query,
not just how topically related it is in general. That's more accurate and
much more expensive per comparison, which is why the standard pattern is:
cheap vector search retrieves a wide candidate set, then the reranker
sorts just those down to the real top-k.

```
cd 05-reranking
../.venv/Scripts/python.exe rag.py "your question" --top-k 3                              # vector only
../.venv/Scripts/python.exe rag.py "your question" --top-k 3 --rerank --retrieve-k 10      # retrieve 10, rerank to 3
```

Reranking here uses Voyage's `rerank-2.5` model via `client.rerank()` —
not a locally-run cross-encoder (which would need `sentence-transformers` +
`torch`, both heavy dependencies) — so this project has no new install
beyond what 01 already set up.

## A real case where vector search's #1 pick isn't the answer

```
../.venv/Scripts/python.exe rag.py "Is my voice data ever shared with other Solstice Labs customers?" --top-k 3
```
Real output — vector search alone, top 3:
```
[1] (score: 0.5647) "Voice commands are processed locally on the Hub by default..."
[2] (score: 0.5608) "Camera and sensor data ... Solstice Labs does not sell customer data to third parties."
[3] (score: 0.5384) "...Clips are never used to train models shared across other customers' accounts..."
```
The chunk that actually answers the question — "Clips are never used to
train models shared across other customers' accounts" — ranks **#3**, behind
two chunks that are topically related (both about privacy/data handling)
but don't answer *this specific* question. At `--top-k 1`, plain vector
search would hand back chunk [1], which never addresses cross-customer
sharing at all.

## Reranking fixes it

```
../.venv/Scripts/python.exe rag.py "Is my voice data ever shared with other Solstice Labs customers?" --top-k 3 --rerank --retrieve-k 10
```
```
[1] (score: 0.7734) "...Clips are never used to train models shared across other customers' accounts..."
[2] (score: 0.7148) "Voice commands are processed locally on the Hub by default..."
[3] (score: 0.6133) "Camera and sensor data ... Solstice Labs does not sell customer data..."
```
The specific-answer chunk jumps to #1, decisively (0.77 vs. 0.71). The
cross-encoder can read "shared across other customers' accounts" against
the actual query text "shared with other Solstice Labs customers" and
recognize that as a much closer match than a chunk that's merely
*about the same topic* — a distinction cosine similarity over independent
embeddings is structurally worse at making.

## What reranking can't fix

Reranking only reorders whatever vector search already retrieved. It has
no way to promote a chunk that never made it into the candidate set in the
first place — `--retrieve-k` sets a hard ceiling on what reranking can
possibly find. On a 20-chunk corpus this rarely bites (`--retrieve-k 10`
already covers half the corpus), but on a real corpus with thousands of
chunks, a `--retrieve-k` set too low silently caps recall before reranking
ever gets a chance to help. Reranking makes the top of the list more
precise; it can't rescue what retrieval missed. (Project 06's evaluation
harness makes this distinction measurable: recall@k on the retrieval step,
separately from ranking quality within the top-k.)

## Cost trade-off, in real numbers

Reranking adds one API call and real latency on top of the retrieval you
already did. Rather than leave that vague, here's what it actually costs,
measured on this project's real corpus and the privacy-question example
above (`retrieve-k=10`), using Voyage's own token counter and pricing:

| Step | Model | Billed tokens (measured) | Price | Cost |
|---|---|---|---|---|
| Query embedding (always happens) | `voyage-3.5` | 16 | $0.06 / 1M tokens | $0.00000096 |
| Rerank call (only with `--rerank`) | `rerank-2.5` | 969 | $0.05 / 1M tokens | $0.0000485 |

Rerank billing isn't "tokens in the query" — it's
`(query_tokens × num_candidates) + sum(all candidate tokens)`. That's why a
13-token query plus 10 short chunks adds up to 969 billed tokens, not 13.

**That's roughly a 50x higher dollar cost per query for reranking than
plain vector search** — but 50x of a very small number is still small:

| Volume | Vector-only | With reranking (`retrieve-k=10`) |
|---|---|---|
| 1,000 queries | $0.001 | $0.05 |
| 1,000,000 queries | $0.96 | $49 |

Voyage also gives every account 200 million free rerank tokens. At ~969
tokens/query here, that's roughly **206,000 reranked queries before you
pay Voyage anything for reranking at all.**

**Latency is the cost that actually bites first.** Timed on real calls:

| Step | Measured time |
|---|---|
| Query embedding (network round-trip) | 384 ms |
| Local vector search (numpy, in-process) | 0.26 ms |
| Rerank API call (network round-trip) | 152 ms |

Vector search itself is free and effectively instant — it's local matrix
math. The two *network calls* are what cost anything, and reranking adds a
second sequential round-trip on top of the first, roughly doubling the
API-call portion of total latency (384ms → 536ms). That's noticeable in a
user-facing app, and compounds fast if you're chaining LLM calls after it.

**The knob that actually controls both costs is `--retrieve-k`, not
`--top-k`.** Both the dollar cost and the latency scale with how many
candidates get reranked — going from `retrieve-k=10` to `retrieve-k=100`
roughly 10x's the rerank bill (both terms of the billing formula scale
with candidate count) and adds proportionally more latency, since the
cross-encoder is scoring 10x as many (query, chunk) pairs. "Retrieve wide"
is not free to make wider.

It's worth it when precision at the very top matters — e.g. you're only
showing the user one answer, or only handing the LLM 3 chunks instead of
10 — and not worth it if you're already passing a generous number of
chunks to a capable LLM that can sort out relevance itself from context.
