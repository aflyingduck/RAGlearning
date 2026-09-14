# 02 — Chunking Strategies

01-naive-rag used fixed-size character chunking and showed it cutting a
word in half at a chunk boundary ("miss**ing**." split across two chunks).
This project builds three better strategies and runs the *same* corpus and
queries through all four, so you can see exactly what each one buys you —
and where each one still falls short.

Run any query through any strategy:
```
cd 02-chunking-strategies
../.venv/Scripts/python.exe rag.py "your question" --strategy fixed|sentence|recursive|semantic
../.venv/Scripts/python.exe rag.py "x" --strategy recursive --dump-chunks   # inspect chunk boundaries directly, no retrieval
```

## The four strategies (`chunkers.py`)

1. **`fixed`** — 01's baseline. Cuts every 500 characters, no awareness of
   words, sentences, or structure at all.
2. **`sentence`** — splits into sentences first (regex on `. `, `! `, `? `
   before a capital letter or digit), then greedily packs whole sentences
   into ~500-char chunks. Never cuts a word, but doesn't know a markdown
   heading means "new topic," so it can still merge unrelated sentences
   from different sections together.
3. **`recursive`** — splits on the *largest* structural boundary first
   (blank lines = paragraphs/sections in markdown), and only falls back to
   sentence-level splitting if a single paragraph is too big to fit in one
   chunk. This is what production text splitters (e.g. LangChain's
   `RecursiveCharacterTextSplitter`) actually do, and it's the default here.
4. **`semantic`** — embeds every sentence, then breaks between two
   consecutive sentences only where their similarity is *lower than most
   other consecutive pairs in that document* — i.e. a bigger topic jump
   than usual. Chunk boundaries follow meaning rather than a
   character/paragraph count at all.

## Fixed vs. recursive: the mid-word cut is gone

Same query that broke 01:
```
../.venv/Scripts/python.exe rag.py "What does the warranty cover and how do I file a claim?" --strategy recursive --top-k 2
```
Top result now:
```
To start a return or warranty claim, go to solsticelabs.com/support and
enter your order number. Warranty claims are typically processed within 5
business days; Solstice Labs will ship a replacement before requiring the
faulty unit back, using a prepaid return label.

The optional Nimbus Battery Base has a separate 1-year warranty, since it is
sold as an accessory rather than part of the core Hub purchase.
```
Two complete, self-contained paragraphs — no orphaned `ing.` fragment. Every
chunk in this doc now starts and ends on a real sentence boundary (verify
with `--dump-chunks`). Paragraph-first splitting fixes exactly the failure
mode 01 surfaced.

## Where recursive still has a rough edge

`--dump-chunks --strategy recursive` on `installation-guide.md` shows the
numbered setup list got pulled apart awkwardly:
```
--- chunk 1 ---
1. Plug the Hub into power using the included USB-C adapter. The status ring
   will pulse blue while it boots, which takes about 30 seconds. 2. Open the Nimbus app...
```
The whole 5-step list is one markdown paragraph (no blank lines between
items) that's too long for one 500-char chunk, so `recursive` falls back to
its sentence splitter — which doesn't know "2." is a list marker, not
sentence-ending punctuation followed by a new sentence. The list survives
without losing content, but its clean numbered structure gets flattened
into a run-on paragraph. Recursive chunking is only as good as the
fallback splitter it reaches for when structure alone isn't enough.

## Where semantic chunking breaks in a more interesting way

`--dump-chunks --strategy semantic` on the same file is worse, not better:
```
--- chunk 1 (installation-guide.md, 176 chars) ---
# Nimbus Hub — Installation Guide
1. Plug the Hub into power using the included USB-C adapter. ...about 30 seconds.

--- chunk 2 (installation-guide.md, 2 chars) ---
2.

--- chunk 3 (installation-guide.md, 221 chars) ---
Open the Nimbus app (iOS 15+ or Android 11+) and tap "Add Hub." ...
```
The same "2." list-marker-as-sentence bug from `recursive` shows up here
too (both strategies share the same sentence splitter) — but semantic
chunking makes it *worse*: a 2-character "sentence" like `"2."` embeds to
an outlier vector with low similarity to its neighbors on both sides, so
the algorithm reads that as a topic change and gives `"2."` its own
standalone chunk. A chunk containing only the text `"2."` is retrievable
by nothing and useful for nothing.

The lesson isn't "semantic chunking is bad" — it's that semantic chunking
inherits every flaw in whatever splits it into sentences first. A fancier
merge strategy on top of a naive sentence splitter is still bounded by that
splitter's mistakes. In production, semantic chunking is usually paired
with a much more robust sentence tokenizer (e.g. spaCy's, or one that
understands markdown/list syntax) — worth remembering before reaching for
"semantic" as an automatic upgrade over "recursive."

## `sentence` chunking: fixes words, not topics

`--dump-chunks --strategy sentence` (no API call needed — chunking itself
is pure text processing) confirms the claim above directly. In
`privacy-and-security.md`, chunk 4 reads:
```
Clips are never used to train models shared across other customers'
accounts. You can delete stored clips immediately from Settings > Privacy >
Manage Voice Data. The Hub encrypts all local network traffic between
itself and paired devices using per-device keys established during
pairing. Hub-to-cloud traffic (for Nimbus Plus features) uses TLS 1.3.
```
The first two sentences are about voice-clip privacy; the last two are
about network encryption — two different paragraphs in the source doc,
fused into one chunk because `sentence` packs by character budget alone
with no concept of "these came from different sections." `recursive`
doesn't make this mistake because it respects the blank-line paragraph
break first.

## Head-to-head: `"What does the warranty cover and how do I file a claim?"`

| Strategy | Top chunk (abridged) | Real defect observed |
|---|---|---|
| `fixed` (01's baseline) | `"ing.\n\nTo start a return or warranty claim, go to solsticelabs.com/support..."` | Cuts mid-word ("miss**ing**.") at the chunk boundary |
| `sentence` | packs whole sentences, but merges across section/paragraph breaks | No mid-word cuts, but topic drift within a chunk (see privacy example above) |
| `recursive` | `"To start a return or warranty claim... The optional Nimbus Battery Base has a separate 1-year warranty..."` | Clean paragraph boundaries; only rough edge is markdown lists losing structure in the sentence-splitter fallback |
| `semantic` | `"The Nimbus Hub 3 ships with a 2-year limited hardware warranty covering manufacturing defects..."` (similarity 0.511) then a chunk merging the return-window and claim-filing sentences together (0.499) | Best topical grouping on prose; but degenerates into junk 2-character chunks (`"2."`, `"4."`) on the numbered installation list — see above |

**Takeaway:** there's no strategy that's unconditionally best. `recursive`
is the safest default for structured markdown/docs (best boundary
correctness, cheapest to compute — one API call for chunk embeddings, no
sentence-level embedding pass). `semantic` groups prose by *meaning* better
than any character-count-based method, but its quality is capped by
sentence-segmentation quality, and it costs an extra embedding call per
corpus (one for sentences, one for final chunks). `fixed` should be treated
as a last resort or a quick prototype baseline, not a production choice.

Project 06 (evaluation) will replace this "read the output and judge"
approach with actual recall@k / MRR numbers across strategies — this
project is about *seeing* the differences; that one is about *measuring*
them.
