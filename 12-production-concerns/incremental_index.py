"""
Incremental indexing: re-embed only the documents that actually changed,
not the whole corpus, every time the index is rebuilt.

Every prior project's embed_chunks() checked ONE hash for the entire
corpus -- change any doc, and the whole corpus re-embeds. That's fine at
20 chunks. 03-vector-stores-and-ann showed index build time growing
non-trivially with corpus size (25ms at 1K vectors, 19 SECONDS at 50K) --
at real-world scale, re-embedding an unchanged 50,000-chunk corpus because
one document changed is a real, avoidable cost. This project hashes
per-document, not per-corpus, so an edit to one doc only re-embeds that
doc's chunks.

Usage:
    python incremental_index.py            # build or update the index
    python incremental_index.py --stats    # show what's in the index without changing it
"""

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import voyageai
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

HERE = Path(__file__).parent
DOCS_DIR = HERE / "data" / "docs"
INDEX_FILE = HERE / "data" / "index_state.json"
EMBED_MODEL = "voyage-3.5"
CHUNK_SIZE = 500


def chunk_recursive(text, doc_name, chunk_size=CHUNK_SIZE):
    sentence_re = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

    def split_sentences(t):
        t = t.strip()
        return [s.strip() for s in sentence_re.split(t) if s.strip()] if t else []

    def pack_sentences(t):
        chunks, current = [], ""
        for sentence in split_sentences(t):
            candidate = f"{current} {sentence}".strip() if current else sentence
            if len(candidate) > chunk_size and current:
                chunks.append(current)
                current = sentence
            else:
                current = candidate
        if current:
            chunks.append(current)
        return chunks

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, current = [], ""
    for para in paragraphs:
        if len(para) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(pack_sentences(para))
            continue
        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) > chunk_size and current:
            chunks.append(current)
            current = para
        else:
            current = candidate
    if current:
        chunks.append(current)
    return [{"doc": doc_name, "text": t} for t in chunks]


def doc_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_index():
    if INDEX_FILE.exists():
        return json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    return {}


def save_index(index):
    INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    INDEX_FILE.write_text(json.dumps(index), encoding="utf-8")


def update_index(client, index):
    """Returns (updated_index, doc_names_reembedded, doc_names_reused, doc_names_removed)."""
    current_docs = {p.name: p.read_text(encoding="utf-8") for p in sorted(DOCS_DIR.glob("*.md"))}

    reembedded, reused = [], []
    changed_docs = {}  # doc_name -> (text, hash) for docs needing embedding

    for name, text in current_docs.items():
        h = doc_hash(text)
        if name in index and index[name]["hash"] == h:
            reused.append(name)
        else:
            changed_docs[name] = (text, h)

    removed = [name for name in index if name not in current_docs]

    if changed_docs:
        # Batch every changed doc's chunks into ONE embed call, regardless
        # of how many docs changed -- still far cheaper than re-embedding
        # the untouched docs too.
        all_new_chunks = []
        chunk_owner = []  # parallel list: which doc each chunk in all_new_chunks belongs to
        for name, (text, _h) in changed_docs.items():
            chunks = chunk_recursive(text, name)
            all_new_chunks.extend(chunks)
            chunk_owner.extend([name] * len(chunks))

        result = client.embed([c["text"] for c in all_new_chunks], model=EMBED_MODEL, input_type="document")
        embeddings = result.embeddings

        per_doc_chunks = {name: [] for name in changed_docs}
        for chunk, emb, owner in zip(all_new_chunks, embeddings, chunk_owner):
            per_doc_chunks[owner].append({"text": chunk["text"], "embedding": emb})

        for name, (text, h) in changed_docs.items():
            index[name] = {"hash": h, "chunks": per_doc_chunks[name]}
            reembedded.append(name)

    for name in removed:
        del index[name]

    return index, reembedded, reused, removed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()

    index = load_index()

    if args.stats:
        total_chunks = sum(len(v["chunks"]) for v in index.values())
        print(f"Indexed docs: {len(index)}, total chunks: {total_chunks}")
        for name, v in index.items():
            print(f"  {name}: {len(v['chunks'])} chunks, hash {v['hash'][:12]}...")
        return

    load_dotenv(dotenv_path=HERE.parent / ".env")
    client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])

    index, reembedded, reused, removed = update_index(client, index)
    save_index(index)

    print(f"Re-embedded ({len(reembedded)}): {reembedded}")
    print(f"Reused from cache ({len(reused)}): {reused}")
    if removed:
        print(f"Removed ({len(removed)}): {removed}")
    total_chunks = sum(len(v["chunks"]) for v in index.values())
    print(f"\nIndex now has {len(index)} docs, {total_chunks} chunks total.")
    print(f"This run made {'1' if reembedded else '0'} embedding API call"
          f"{'s' if len(reembedded) != 1 else ''} (batched across all changed docs), "
          f"regardless of corpus size.")


if __name__ == "__main__":
    main()
