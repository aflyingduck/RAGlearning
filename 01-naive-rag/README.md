# 01 — Naive RAG

The simplest version of RAG that still counts as RAG. Every later project in
this curriculum is "naive RAG, but we fixed one specific problem with it" —
so understanding exactly what this one gets wrong is the point.

## The four concepts

**Chunking.** Documents are usually too big to hand to a search step or an
LLM context window wholesale, so we cut them into smaller pieces first. Here
we use *fixed-size character chunking*: cut every 500 characters, with a
50-character overlap between consecutive chunks so a fact sitting right at a
cut point has a chance of surviving in one of the two chunks. This is the
crudest chunking strategy that exists — it doesn't know what a sentence,
paragraph, or heading is. See "Where this breaks," below.

**Embedding.** An embedding model turns text into a vector of numbers (here,
1024 of them, from Voyage's `voyage-3.5` model) positioned so that
*semantically similar* text ends up at nearby points in that 1024-dimensional
space. "How do I reset the hub" and "steps to factory reset" land near each
other even though they don't share many words — that's what makes semantic
search different from keyword search (grep).

Note this is a *separate model* from the LLM that eventually generates an
answer. Claude doesn't produce embeddings; Voyage AI does. Every RAG system
has (at least) two models in it: one that indexes/retrieves, one that
generates. They don't have to be from the same vendor, and in this project
they aren't.

**Vector search.** Once every chunk and the query are both points in the
same embedding space, "find the most relevant chunks" becomes "find the
chunks whose vectors are closest to the query's vector." We measure
closeness with *cosine similarity* — the cosine of the angle between two
vectors, ranging from -1 (opposite) to 1 (identical direction). It's
computed here as plain numpy (`rag.py`'s `cosine_similarity` function) rather
than a vector database, so you can see that "search" is really just a dot
product over normalized vectors. At this corpus size (19 chunks) that's
instant; real corpora need approximate nearest-neighbor indexes, which is a
later topic.

**Retrieval (not generation).** `rag.py` stops here. It prints the top-k
chunks and your query in a delimited block — it does not call an LLM. Paste
that block into this chat and I'll generate the answer using only what was
retrieved. This is deliberate: it makes the retrieval step's output visible
and gradeable on its own, separate from how good the LLM is at writing.
Almost all practical RAG quality problems are retrieval problems wearing a
generation-problem disguise.

## Try it

```
cd 01-naive-rag
../.venv/Scripts/python.exe rag.py "How long is the warranty on the Hub 3?"
```

The corpus (`data/docs/`) is fictional product documentation for a made-up
smart-home hub — fictional on purpose, so that if an answer is right, you
know it came from retrieval and not from something Claude already knew about
a real product.

First run embeds all chunks via the Voyage API and caches the vectors to
`data/cache/embeddings.json` (gitignored); later runs only re-embed if you
edit a doc or change `CHUNK_SIZE`/`CHUNK_OVERLAP` in `rag.py`. Only the query
gets embedded fresh each time.

Useful flags:
- `--top-k N` — how many chunks to retrieve (default 3)
- `--show-scores` — dump every chunk's similarity score to stderr, sorted,
  so you can see how close the non-retrieved chunks were

## Worked example: retrieval to answer

Retrieved context for `"What does the warranty cover and how do I file a claim?"`
(two chunks, both from `warranty-and-returns.md`) grounds this answer:

> The Nimbus Hub 3 has a 2-year limited hardware warranty covering
> manufacturing defects; it does not cover physical damage, water damage, or
> issues caused by third-party power adapters. To file a claim, go to
> solsticelabs.com/support and enter your order number — claims are
> typically processed within 5 business days, and Solstice Labs ships a
> replacement before requiring the faulty unit back, using a prepaid return
> label.

Now try a question the corpus has no answer for:
`"What colors does the Hub come in?"` The correct response is a refusal —
*"I can't answer this from the provided context — it doesn't mention Hub
color options"* — not a guess. Two tells that the retrieval failed to find
anything relevant: the answer text doesn't actually mention color, and (run
with `--show-scores`) every chunk's similarity score sits close together in
a narrow band (0.51 down to 0.29) instead of one chunk clearly leading the
pack the way the warranty query's top hit did. A flat score distribution is
itself a signal worth watching for, separate from reading the retrieved
text — it's how you'd programmatically catch "nothing relevant was found"
before generation ever happens.

## Where this breaks

Run:
```
../.venv/Scripts/python.exe rag.py "What does the warranty cover and how do I file a claim?" --top-k 2
```

The top result starts like this:

```
ing.

To start a return or warranty claim, go to solsticelabs.com/support and
...
```

That `ing.` is the tail end of the word "**miss**ing." — fixed-size chunking
cut a sentence about restocking fees in half mid-word, and the 50-character
overlap window happened to start right there. The chunker has no idea a word
boundary, sentence, or heading exists; it only counts characters. On this
tiny, clean corpus the damage is cosmetic (the answer is still buried in
there and retrievable). On a real corpus — especially one with tables, code
blocks, or short list items — the same mechanism routinely severs a fact
from the sentence that gives it meaning, so *neither* resulting chunk embeds
close to a query about that fact, and it's never retrieved at all.

Two other things to notice while you're here:
- **`--show-scores`** on a query like `"how much power does the hub use"`
  shows the right chunk winning, but the margin over unrelated chunks is
  often smaller than you'd expect — cosine similarity over generic
  embeddings is a blunter instrument than "semantic search" sounds like.
- **Rate limits are real.** The Voyage free tier without a payment method on
  file is capped at 3 requests/minute. If you script a loop of test queries,
  you'll hit `RateLimitError` fast — worth knowing before you assume a
  production pipeline "just works" at whatever query volume you throw at it.

These two problems — chunking that ignores document structure, and
similarity search with no way to weight exact keyword matches — are exactly
what projects 02 and 04 fix.
