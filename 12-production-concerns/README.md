# 12 — Production Concerns

Two things every project so far quietly assumed away: that the whole
corpus gets re-embedded from scratch every time (fine at 20 chunks, not
fine at scale — see 03's 19-second index build at 50K vectors), and that
retrieval quality, once verified, stays verified. Both assumptions break
in a real, running system. This project addresses both.

## Incremental indexing

`incremental_index.py` hashes each DOCUMENT individually (not the whole
corpus as one hash, like every prior project's cache) and only re-embeds
documents whose content actually changed.

```
cd 12-production-concerns
../.venv/Scripts/python.exe incremental_index.py            # build or update
../.venv/Scripts/python.exe incremental_index.py --stats    # inspect without changing anything
```

Verified real behavior, three runs:
1. **First run** (empty index): `Re-embedded (6): [...all 6 docs...]` — 1
   batched API call for the whole corpus, same as every other project.
2. **Second run**, no changes: `Re-embedded (0): []` — **zero API calls**,
   every doc's hash matched and was reused straight from the index file.
3. **Third run**, after editing `subscription-plans.md` only:
   `Re-embedded (1): ['subscription-plans.md']` — the other 5 docs were
   untouched and cost nothing to "re-check."

At this corpus size the savings are trivial. At 03's 50,000-chunk scale
(19-second brute rebuild), re-embedding and re-indexing the other 49,990
unchanged chunks every time one document changes is a real, recurring cost
this pattern eliminates.

## Retrieval-quality drift monitoring

`drift_monitor.py` runs a small fixed benchmark (4 labeled queries, same
idea as 06's eval harness) against the current index, logs the score
distribution with a timestamp to `data/drift_log.jsonl`, and flags
low-confidence results and wrong-document top hits.

```
../.venv/Scripts/python.exe drift_monitor.py            # run + log
../.venv/Scripts/python.exe drift_monitor.py --history   # show all logged runs
```

### A real incident, simulated and caught

To test whether this actually catches something, `troubleshooting.md` was
deliberately truncated down to just its heading — simulating a real
ingestion bug (a parser that silently truncates a doc, a bad deploy that
ships an empty file, a CMS export that drops content). Real before/after:

```
2026-07-19T18:57:10Z: mean_top1=0.688  min_top1=0.608  low_confidence=0/4  wrong_doc=0/4
2026-07-19T18:57:48Z: mean_top1=0.614  min_top1=0.471  low_confidence=1/4  wrong_doc=1/4
```

Per-query detail on the second run:
```
  0.555  [installation-guide.md] 'How do I factory reset the Hub?'  <-- WRONG DOC
  0.471  [troubleshooting.md] "What should I do if a device shows online but won't respond?"
  0.711  [warranty-and-returns.md] 'How long is the warranty on the Hub 3?'
  0.721  [installation-guide.md] 'What Wi-Fi band does the Hub need for initial setup?'
```

Both queries that depend on `troubleshooting.md` degraded — one dropped
below the low-confidence threshold, the other returned the **wrong
document entirely** once its real content was gone. The two queries that
depend on unrelated docs (warranty, installation) scored *identically* to
the healthy baseline (0.711, 0.721 both times) — confirming incremental
indexing correctly isolated the change to just the broken document,
and the monitor correctly isolated the regression to just the affected
queries instead of reporting a vague overall dip.

Without this, that incident would have been invisible: the corpus "has 6
files," the pipeline "ran successfully," nothing errors — the only symptom
is that specific answers quietly get worse or wrong, which nobody notices
until a user complains. A benchmark that runs and logs every time the
index updates turns that into a number that moved, on a specific query,
right after a specific change.

### What this monitor doesn't catch

A low top-1 score is a reasonable proxy for "something's wrong," but it's
not the only failure mode. A chunking change that still scores confidently
but retrieves the wrong chunk (03/06 both saw ranking differences that
didn't come with a big score change) wouldn't necessarily trip the
`LOW_CONFIDENCE_THRESHOLD` here — the labeled recall/MRR checks from 06
are the right tool for that, run periodically the same way. Score-based
drift monitoring and labeled-eval regression testing answer different
questions ("did anything get vaguely worse" vs. "is this specific known
case still right") and a real production setup wants both, not one
instead of the other.
