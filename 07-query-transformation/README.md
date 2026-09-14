# 07 — Query Transformation

Every prior project transformed the CORPUS (chunking, indexing, reranking).
This one transforms the QUERY before searching with it. Two techniques:

- **Multi-query**: split one question into several narrower sub-questions,
  retrieve for each, and fuse the rankings (RRF, same as 04).
- **HyDE** (Hypothetical Document Embeddings): generate a plausible fake
  *answer* to the query, and embed that instead of the query itself —
  betting that a real answer chunk resembles another answer-shaped piece
  of text more than it resembles a question.

Both techniques require generating text (sub-questions, or a hypothetical
answer) — a generation task, so per this curriculum's pattern that
generation happens here, in chat, not via an API call inside the script.
`rag.py` takes the already-generated sub-queries or hypothetical doc as CLI
arguments:

```
cd 07-query-transformation
../.venv/Scripts/python.exe rag.py "query" --mode baseline
../.venv/Scripts/python.exe rag.py "query" --mode multiquery --sub-queries '["sub q 1", "sub q 2"]'
../.venv/Scripts/python.exe rag.py "query" --mode hyde --hyde-doc "a plausible hypothetical answer..."
```

**If you're in Windows PowerShell (not bash/WSL), `--sub-queries` needs
different quoting than shown above.** PowerShell wraps any argument
containing spaces in its own outer double quotes when handing it to a
native `.exe`, but it does *not* escape double-quote characters already
inside that argument — and JSON needs literal `"` characters. The result is
PowerShell's quoting collides with JSON's quoting, and argparse sees the
value torn apart at every space (`unrecognized arguments: ...`). Use a
double-quoted PowerShell string with every internal `"` escaped as `` \`" ``
(backslash + backtick + quote) instead of the single-quoted form above:

```powershell
python.exe rag.py "query" --mode multiquery --top-k 5 --sub-queries "[\`"sub q 1\`", \`"sub q 2\`"]"
```

Same applies to `--hyde-doc` only if the hypothetical-answer text itself
contains a `"` character.

## First finding: simple paraphrase mostly doesn't break voyage-3.5

Before building anything, this project tested whether colloquial rephrasing
("my gadget won't behave," "think of it like a walkie talkie that stopped
transmitting") throws vector search off the scent of the corpus. It mostly
doesn't — modern embedding models are considerably more robust to
paraphrase and metaphor than older tutorials on RAG (which often predate
today's embedding models) suggest. Several candidate "vector search should
obviously fail here" queries were tried and rejected for this project
specifically because the baseline handled them fine. That's worth knowing
on its own: don't reach for query transformation to fix a paraphrase
problem you haven't confirmed you actually have.

## Where a real weakness showed up: compound questions

```
../.venv/Scripts/python.exe rag.py "I have a huge house and one of these little boxes probably can't cover it all -- what are my options?" --mode baseline --top-k 5
```
Real baseline output, top 3:
```
[1] (0.3487) "For Hub Mesh setups, install the primary Hub first..."
[2] (0.3474) "The Hub 3 supports up to 200 connected devices per hub. Larger homes can pair multiple hubs..."
[3] (0.3467) "Nimbus Plus does not change local automation speed... Family plans covering up to 5 hubs..."
```
[1] and [2] are genuinely useful (device limit + how to link hubs). [3] is
about subscription pricing tiers, not physical coverage — related to "5
hubs" by coincidence of vocabulary, not by actually answering the
question. This is a compound question (implicitly: "what's the device
limit?" + "how do I add more hubs?") represented as a single embedding,
which has to compress two distinct facts into one vector and inevitably
loses precision on both.

## Multi-query made it WORSE — a real, useful negative result

```
../.venv/Scripts/python.exe rag.py "..." --mode multiquery --top-k 5 \
  --sub-queries '["What is the maximum number of devices a single Nimbus Hub supports?", "How do I connect multiple Nimbus Hubs together to cover a larger home?"]'
```
(On PowerShell, use the `\`"`-escaped double-quoted form from the usage
section above instead of the single-quoted form shown here.)

Real output, top 5 — **the directly relevant Hub Mesh setup-steps chunk
(baseline's #1) drops out of the top 5 entirely**, replaced by generic
installation steps and the same marginal subscription-pricing chunk from
before. RRF fusion across only two rankers, on a 20-chunk corpus, turned
out to dilute a strong single-ranker signal rather than sharpen it. This
wasn't cherry-picked to make a point — it's the actual first result, and
it's a legitimate finding: **query transformation is not a free win.** It
costs an extra generation step, extra API calls, and here it measurably
hurt. Multi-query tends to help more on much larger corpora and with more
sub-queries/deduping logic than a bare 2-way RRF fusion; at this scale, the
fusion overhead wasn't earning its keep.

## HyDE was a clean, decisive win

```
../.venv/Scripts/python.exe rag.py "..." --mode hyde --top-k 5 \
  --hyde-doc "If your home is too large for a single Hub, you can add additional Nimbus Hubs and link them together using Hub Mesh, which lets automations reference devices on any paired hub. Each Hub supports up to 200 connected devices, so pairing multiple hubs increases your home's total device capacity and coverage for a larger house."
```
Real output, top 3:
```
[1] (0.8848) "The Hub 3 supports up to 200 connected devices per hub. Larger homes can pair multiple hubs..."
[2] (0.8622) "For Hub Mesh setups, install the primary Hub first..."
[3] (0.8199) "Hub Mesh shows a paired hub as offline..." (troubleshooting, still on-topic)
```
Both genuinely relevant chunks now occupy #1 and #2, and the marginal
subscription-pricing chunk is gone from the top 3. Also notice the
absolute similarity scores: **0.88/0.86/0.82, versus 0.35/0.35/0.35 for the
same corpus under baseline.** That jump isn't really about relevance
ranking — it's because the HyDE text is written in the same register as
the corpus (declarative, document-shaped prose) rather than as a colloquial
question, so it sits much closer to real chunks in embedding space
regardless of content. Don't compare HyDE similarity scores to raw-query
similarity scores as if they're the same scale — they aren't.

## Takeaways

- Confirm a real weakness before reaching for either technique — most of
  this project's effort went into finding a query where baseline actually
  struggled, because most colloquial rephrasings didn't break it.
- HyDE helped here because the compound question's *implicit facts* (device
  limit, hub-linking) are things a hypothetical answer states explicitly
  and declaratively — closer in form to how the corpus itself is written.
- Multi-query's fusion step can hurt as easily as help; it needs either
  more sub-queries, a larger corpus, or a smarter merge than plain 2-way
  RRF to reliably pay for its own cost. Measure before shipping it (06's
  eval harness is the right tool for checking this on your own queries
  rather than trusting either this README or intuition).
