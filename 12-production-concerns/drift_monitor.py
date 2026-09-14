"""
Retrieval-quality drift monitor: run a fixed benchmark query set against
the current index, log the score distribution, and compare against
history -- the same idea as 06's evaluation harness, but run repeatedly
over time instead of once, so a regression shows up as a trend instead of
requiring someone to notice it by eye.

This won't catch every regression (a bad chunking change that still scores
confidently, just wrongly, needs 06-style labeled recall/MRR checks, not
just a score-magnitude monitor) -- but it catches a real, common class of
production incident cheaply: something broke ingestion for a document
(truncation, encoding corruption, a doc silently going empty) and that
document's queries all get quietly worse, invisibly, until someone
notices.

Usage:
    python drift_monitor.py            # run the benchmark, log + print results
    python drift_monitor.py --history  # show all logged runs
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import voyageai
from dotenv import load_dotenv

from incremental_index import load_index, update_index, save_index

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

HERE = Path(__file__).parent
LOG_FILE = HERE / "data" / "drift_log.jsonl"
EMBED_MODEL = "voyage-3.5"
LOW_CONFIDENCE_THRESHOLD = 0.5

BENCHMARK_QUERIES = [
    {"query": "How do I factory reset the Hub?", "expected_doc": "troubleshooting.md"},
    {"query": "What should I do if a device shows online but won't respond?", "expected_doc": "troubleshooting.md"},
    {"query": "How long is the warranty on the Hub 3?", "expected_doc": "warranty-and-returns.md"},
    {"query": "What Wi-Fi band does the Hub need for initial setup?", "expected_doc": "installation-guide.md"},
]


def flatten_index(index):
    chunks, vecs = [], []
    for doc_name, entry in index.items():
        for c in entry["chunks"]:
            chunks.append({"doc": doc_name, "text": c["text"]})
            vecs.append(c["embedding"])
    return chunks, np.array(vecs, dtype=np.float32)


def run_benchmark(client, index):
    chunks, vecs = flatten_index(index)
    chunk_norm = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    queries = [q["query"] for q in BENCHMARK_QUERIES]
    result = client.embed(queries, model=EMBED_MODEL, input_type="query")

    results = []
    for item, qvec in zip(BENCHMARK_QUERIES, result.embeddings):
        qn = np.array(qvec, dtype=np.float32)
        qn = qn / np.linalg.norm(qn)
        scores = chunk_norm @ qn
        top_i = int(np.argmax(scores))
        results.append({
            "query": item["query"],
            "expected_doc": item["expected_doc"],
            "top_doc": chunks[top_i]["doc"],
            "top_score": float(scores[top_i]),
            "correct_doc": chunks[top_i]["doc"] == item["expected_doc"],
        })
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", action="store_true")
    args = parser.parse_args()

    if args.history:
        if not LOG_FILE.exists():
            print("No history yet.")
            return
        for line in LOG_FILE.read_text(encoding="utf-8").splitlines():
            run = json.loads(line)
            print(f"{run['timestamp']}: mean_top1={run['mean_top1_score']:.3f}  "
                  f"min_top1={run['min_top1_score']:.3f}  "
                  f"low_confidence={run['low_confidence_count']}/{len(BENCHMARK_QUERIES)}  "
                  f"wrong_doc={run['wrong_doc_count']}/{len(BENCHMARK_QUERIES)}")
        return

    load_dotenv(dotenv_path=HERE.parent / ".env")
    client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])

    index = load_index()
    index, reembedded, reused, removed = update_index(client, index)
    save_index(index)
    if reembedded:
        print(f"(index updated -- re-embedded: {reembedded})\n", file=sys.stderr)

    results = run_benchmark(client, index)

    scores = [r["top_score"] for r in results]
    run_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mean_top1_score": float(np.mean(scores)),
        "min_top1_score": float(np.min(scores)),
        "low_confidence_count": sum(1 for s in scores if s < LOW_CONFIDENCE_THRESHOLD),
        "wrong_doc_count": sum(1 for r in results if not r["correct_doc"]),
        "per_query": results,
    }

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(run_record) + "\n")

    print(f"mean top-1 score: {run_record['mean_top1_score']:.3f}")
    print(f"min top-1 score:  {run_record['min_top1_score']:.3f}")
    print(f"low-confidence queries (score < {LOW_CONFIDENCE_THRESHOLD}): {run_record['low_confidence_count']}/{len(results)}")
    print(f"wrong top doc: {run_record['wrong_doc_count']}/{len(results)}\n")
    for r in results:
        flag = "" if r["correct_doc"] else "  <-- WRONG DOC"
        print(f"  {r['top_score']:.3f}  [{r['top_doc']}] {r['query']!r}{flag}")


if __name__ == "__main__":
    main()
