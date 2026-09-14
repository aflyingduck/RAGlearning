# 09 — Agentic RAG

Every prior project runs retrieval exactly once per query, then hands the
result to generation. Agentic RAG lets the model driving the process
(Claude, here, calling `search.py` directly via its own tool use — no new
framework needed) decide whether one search was enough, and issue further,
differently-worded searches when it wasn't — the same way you'd dig deeper
on a real research question instead of accepting the first answer you find.

`search.py` is deliberately the same single-shot retrieval pipeline as
01/02 — nothing about the *search* changed. What's different is who calls
it, how many times, and why.

```
cd 09-agentic-rag
../.venv/Scripts/python.exe search.py "your question" --top-k 3
```

## The question

*"I have 250 devices across my home — what will I need and what will it
cost?"*

Answering this requires combining three separate facts (device limit,
Hub Mesh pairing, subscription pricing) AND doing arithmetic no retrieval
step can do (250 > 200, so one hub isn't enough) AND checking whether any
constraint might block the obvious plan.

## What single-shot retrieval actually got

```
../.venv/Scripts/python.exe search.py "I have 250 devices across my home -- what will I need and what will it cost?" --top-k 3
```
Real result: top-3 already surfaced all three facts reasonably well — the
200-device-per-hub limit and Hub Mesh pairing (#1), the $8/month family
plan covering up to 5 hubs (#2), and Hub Mesh setup steps (#3). This is
worth stating plainly: **on this corpus, single-shot retrieval usually
isn't the bottleneck** — voyage-3.5 is strong enough that most single
queries already find what's relevant (a pattern this whole curriculum kept
running into, e.g. 07's failed attempts to find a paraphrase that broke
retrieval). So agentic RAG's value here isn't "rescue a failed search" —
it's two other things single-shot structurally *cannot* do:

1. **Do the reasoning.** No retrieval step, however good, computes "250 >
   200, so I need 2 hubs" or "2 hubs costs the same $8/month via the family
   plan as buying Nimbus Plus per-hub." That happens at generation time,
   using the retrieved facts as inputs — which is exactly why this
   curriculum keeps the generation step interactive rather than baking a
   canned answer into the retrieval script.
2. **Notice what it hasn't checked, and go check it.** A single-shot query
   retrieves once and stops, whether or not the results are actually
   sufficient. It has no mechanism to ask "is there a constraint I haven't
   ruled out?"

## The agentic step: verifying a constraint single-shot never thought to check

Before concluding "just add a second hub," a careful agent asks: is there
an upper limit on how many hubs can share one Hub Mesh group, independent
of the 200-devices-per-hub cap? None of the three facts above address
that. So — as the agent, mid-investigation — issue a targeted follow-up
search instead of assuming:

```
../.venv/Scripts/python.exe search.py "maximum number of hubs allowed in a single Hub Mesh group" --top-k 5
```

Real result: nothing in the top 5 states any such cap. The only stated
Hub Mesh constraint is that all paired hubs must share a local network
(troubleshooting.md, #3) — no mention of a hub-count ceiling anywhere in
the corpus.

## The synthesized answer

With both searches done, the honest answer is:

> 250 devices exceeds one Hub's 200-device limit, so you'd need a second
> Hub 3 paired via Hub Mesh. A Nimbus Plus family plan covers up to 5 hubs
> for $8/month — the same price as buying an individual Nimbus Plus
> subscription for one hub ($4/month), so for 2+ hubs the family plan is
> the better deal. **The documentation doesn't specify an upper limit on
> the number of hubs in a single Hub Mesh group** beyond requiring them to
> share a local network — if that matters for your setup, that's worth
> confirming with support rather than assuming either way.

That last sentence — an explicit, correct "I checked and this isn't
documented" — is the actual deliverable of this project. A single-shot
system has no way to produce it: it can only synthesize from what it
happened to retrieve, with no signal about what it never thought to look
for. A single-shot answer built from just the first search's top-3 would
have silently glossed over the constraint question rather than flagging
it — small in this toy example, but the same gap in a real support or
legal or medical corpus is how confident-sounding wrong answers happen.

## Why this wasn't built as a bigger framework

Real agentic RAG systems (e.g. an LLM with a `search` tool it can call
repeatedly) formalize this loop with an explicit stop condition and a
budget on how many searches it's allowed. This project skips building that
scaffolding because Claude, driving `search.py` through its own tool use
in this chat, already *is* that loop — the point being demonstrated is the
decision process (when is one search not enough, and what do you search
for next), not the orchestration code around it.
