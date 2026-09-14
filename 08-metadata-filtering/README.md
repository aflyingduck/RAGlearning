# 08 — Metadata Filtering

Every project so far searched the whole corpus by meaning alone. This one
adds structured metadata to each chunk — which Hub *generation* a doc
applies to — and lets a query restrict the candidate set before vector
search runs at all, instead of hoping embedding similarity keeps
conflicting content out on its own.

The corpus gained a new doc, `hub2-legacy-notes.md`, describing the older
Hub 2 — which shares a lot of vocabulary with the Hub 3 docs ("factory
reset," "device limit," "warranty") but has *different, conflicting*
answers. `data/metadata.json` tags every doc with `{"model": "hub2"}` or
`{"model": "hub3"}`.

```
cd 08-metadata-filtering
../.venv/Scripts/python.exe rag.py "your question" --top-k 3                  # no filter
../.venv/Scripts/python.exe rag.py "your question" --top-k 3 --model hub2     # only hub2-tagged chunks
```

## The real problem: near-tied conflicting instructions

```
../.venv/Scripts/python.exe rag.py "How do I factory reset my Hub 2?" --top-k 3
```
Real output, unfiltered:
```
[1] (hub2-legacy-notes.md, model: hub2, score: 0.7131)
**Hub 2 factory reset.** Hold the pairing button for 10 seconds until the
status ring flashes orange...

[2] (troubleshooting.md, model: hub3, score: 0.7050)
**Factory reset.**
Hold the pairing button for 15 seconds until the status ring flashes purple...
```
The correct chunk technically wins — barely, by 0.0081, well inside normal
noise for this embedding model. **Both chunks make it into the top-3
context handed to the generation step, and they directly contradict each
other** (10s/orange vs. 15s/purple). This is worse than a simple ranking
miss: even when retrieval "works" (the right answer is in the results),
handing an LLM two contradictory procedures with no signal about which
applies is a real failure mode. A slightly different corpus, a slightly
different phrasing, or a slightly different embedding model could easily
flip which one ranks first — nothing here is a big enough margin to trust.

## Filtering fixes it categorically, not probabilistically

```
../.venv/Scripts/python.exe rag.py "How do I factory reset my Hub 2?" --top-k 3 --model hub2
```
```
[1] (hub2-legacy-notes.md, model: hub2, score: 0.7134)  Hub 2 factory reset...
[2] (hub2-legacy-notes.md, model: hub2, score: 0.5720)  # Nimbus Hub 2 — Legacy Notes...
[3] (hub2-legacy-notes.md, model: hub2, score: 0.5661)  Hub 2 hardware...
```
Every result is hub2-tagged. The Hub 3 instructions can't appear at any
score, because filtering happens on the candidate SET before scoring, not
as a post-hoc rank cutoff — an excluded chunk never gets a chance to
outscore an included one, no matter how close the embeddings are. Compare
this to reranking (05): reranking improves ORDERING among retrieved
candidates but can't guarantee exclusion; metadata filtering guarantees
exclusion but has no opinion about ordering within what's left. They solve
different problems and compose fine together.

## The filter has to come from somewhere real

This script takes `--model` as a CLI flag, i.e. the caller already knows
which Hub generation is relevant. In a real system that's the hard part —
either the user states it explicitly ("I have a Hub 2..."), or it has to
be inferred (from account/purchase data, from a previous turn in a
conversation, or by asking a clarifying question). Metadata filtering is
only as good as the metadata extraction feeding it: mistag a doc, or fail
to detect which product version a user has, and you get confident, clean,
*wrong* answers instead of the ambiguous-but-recoverable mix unfiltered
search would have produced. Silently excluding the right answer because a
filter was set incorrectly is a real risk worth weighing against the
contamination risk this project demonstrates.
