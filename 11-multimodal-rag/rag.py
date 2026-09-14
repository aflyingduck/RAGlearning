"""
Multimodal RAG: text chunks and images embedded into the SAME vector
space, searched together.

Two images live only as images here (data/images/) -- their content
(a status-ring color legend, a Hub Mesh topology diagram) is not
duplicated anywhere as text or captions. Voyage's multimodal embedding
model (voyage-multimodal-3) embeds raw images and text into one shared
space, so a query can retrieve an image directly based on its visual
content, the same way it retrieves a text chunk based on its words.

Usage:
    python rag.py "your question" --top-k 3
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
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

HERE = Path(__file__).parent
DOCS_DIR = HERE / "data" / "docs"
IMAGES_DIR = HERE / "data" / "images"
CACHE_FILE = HERE / "data" / "cache" / "embeddings.json"
TEXT_EMBED_MODEL = "voyage-3.5"
MULTIMODAL_MODEL = "voyage-multimodal-3"
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
                chunks.append({"doc": doc_name, "text": current})
                current = ""
            for sub in pack_sentences(para):
                chunks.append({"doc": doc_name, "text": sub})
            continue
        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) > chunk_size and current:
            chunks.append({"doc": doc_name, "text": current})
            current = para
        else:
            current = candidate
    if current:
        chunks.append({"doc": doc_name, "text": current})
    return chunks


def load_items():
    """Returns a unified list of {type, doc, text|image_path}."""
    items = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        for c in chunk_recursive(path.read_text(encoding="utf-8"), path.name):
            items.append({"type": "text", "doc": c["doc"], "text": c["text"]})
    for path in sorted(IMAGES_DIR.glob("*.png")):
        items.append({"type": "image", "doc": path.name, "path": str(path)})
    return items


def embed_items(client, items):
    """Embed text items with voyage-3.5 (matching every other project's
    cache/index) and image items with the multimodal model, in one call
    each. They land in DIFFERENT vector spaces (different models, no
    guaranteed shared geometry) -- see README for how this project handles
    that.
    """
    cache = {}
    if CACHE_FILE.exists():
        cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))

    text_items = [it for it in items if it["type"] == "text"]
    image_items = [it for it in items if it["type"] == "image"]

    text_key = hashlib.sha256(json.dumps([it["text"] for it in text_items]).encode()).hexdigest()
    if cache.get("text", {}).get("key") != text_key:
        result = client.embed([it["text"] for it in text_items], model=TEXT_EMBED_MODEL, input_type="document")
        cache["text"] = {"key": text_key, "embeddings": result.embeddings}

    image_key = hashlib.sha256(json.dumps([it["doc"] for it in image_items]).encode()).hexdigest()
    if cache.get("image", {}).get("key") != image_key:
        images = [Image.open(it["path"]) for it in image_items]
        result = client.multimodal_embed([[img] for img in images], model=MULTIMODAL_MODEL, input_type="document")
        cache["image"] = {"key": image_key, "embeddings": result.embeddings}

    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache), encoding="utf-8")

    text_vecs = np.array(cache["text"]["embeddings"], dtype=np.float32)
    image_vecs = np.array(cache["image"]["embeddings"], dtype=np.float32)
    return text_items, text_vecs, image_items, image_vecs


def rank(query_vec, vecs):
    norm = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    qn = query_vec / np.linalg.norm(query_vec)
    scores = norm @ qn
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    load_dotenv(dotenv_path=HERE.parent / ".env")
    client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])

    items = load_items()
    text_items, text_vecs, image_items, image_vecs = embed_items(client, items)

    text_query = client.embed([args.query], model=TEXT_EMBED_MODEL, input_type="query")
    text_qvec = np.array(text_query.embeddings[0], dtype=np.float32)
    text_scores = rank(text_qvec, text_vecs)

    mm_query = client.multimodal_embed([[args.query]], model=MULTIMODAL_MODEL, input_type="query")
    mm_qvec = np.array(mm_query.embeddings[0], dtype=np.float32)
    image_scores = rank(mm_qvec, image_vecs)

    # Text and image scores come from DIFFERENT models/embedding spaces, so raw
    # cosine similarities aren't directly comparable -- see README. This
    # min-max normalizes each modality's scores to [0, 1] before merging, a
    # simple (imperfect) fix that at least prevents one modality's naturally
    # higher/lower score range from always winning or losing by construction.
    def normalize(scores):
        lo, hi = scores.min(), scores.max()
        return (scores - lo) / (hi - lo) if hi > lo else scores

    combined = (
        [(text_items[i], "text", s) for i, s in enumerate(normalize(text_scores))]
        + [(image_items[i], "image", s) for i, s in enumerate(normalize(image_scores))]
    )
    combined.sort(key=lambda x: -x[2])

    print(f"=== RESULTS for {args.query!r} ===\n")
    for rank_i, (item, kind, score) in enumerate(combined[:args.top_k], 1):
        if kind == "text":
            print(f"[{rank_i}] TEXT (source: {item['doc']}, normalized score: {score:.3f})")
            print(item["text"])
        else:
            print(f"[{rank_i}] IMAGE (source: {item['doc']}, normalized score: {score:.3f})")
            print(f"  -> {item['path']}")
        print()


if __name__ == "__main__":
    main()
