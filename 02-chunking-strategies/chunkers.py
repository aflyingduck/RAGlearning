"""
Four chunking strategies, from crudest to most structure-aware.

01-naive-rag used fixed-size character chunking, which is blind to document
structure and, as its README showed, can cut a fact off mid-word. This file
builds three alternatives so you can compare all four head-to-head on the
same corpus and queries via rag.py --strategy.
"""

import re

import numpy as np

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def split_sentences(text: str):
    """Split on sentence-ending punctuation followed by whitespace and a
    capital letter or digit. Not linguistically rigorous (abbreviations
    like "e.g." will still split), but it never cuts a word in half, which
    is the specific failure this project is fixing.
    """
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in SENTENCE_RE.split(text) if s.strip()]


def chunk_fixed(text: str, doc_name: str, chunk_size=500, overlap=50):
    """Baseline from 01-naive-rag: fixed character count, blind to structure."""
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            chunks.append({"doc": doc_name, "text": chunk, "strategy": "fixed"})
        start += chunk_size - overlap
    return chunks


def chunk_sentence(text: str, doc_name: str, chunk_size=500):
    """Pack whole sentences into chunks up to chunk_size. Never splits a
    sentence mid-word -- but it also doesn't know a markdown heading
    introduces a new topic, so unrelated sentences can still get merged
    across a section boundary.
    """
    sentences = split_sentences(text)
    chunks, current = [], ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) > chunk_size and current:
            chunks.append({"doc": doc_name, "text": current, "strategy": "sentence"})
            current = sentence
        else:
            current = candidate
    if current:
        chunks.append({"doc": doc_name, "text": current, "strategy": "sentence"})
    return chunks


def chunk_recursive(text: str, doc_name: str, chunk_size=500):
    """Structure-aware chunking: split on the largest structural boundary
    first (blank line = paragraph/section), falling back to sentence-level
    splitting only for oversized paragraphs. This is what production
    splitters (e.g. LangChain's RecursiveCharacterTextSplitter) actually do:
    try the most meaningful boundary first, and only break it up further if
    it doesn't fit.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, current = [], ""
    for para in paragraphs:
        if len(para) > chunk_size:
            if current:
                chunks.append({"doc": doc_name, "text": current, "strategy": "recursive"})
                current = ""
            for sub in chunk_sentence(para, doc_name, chunk_size):
                sub["strategy"] = "recursive"
                chunks.append(sub)
            continue
        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) > chunk_size and current:
            chunks.append({"doc": doc_name, "text": current, "strategy": "recursive"})
            current = para
        else:
            current = candidate
    if current:
        chunks.append({"doc": doc_name, "text": current, "strategy": "recursive"})
    return chunks


def chunk_semantic(text: str, doc_name: str, sentence_vecs, threshold_percentile=25):
    """Semantic chunking: given a document's sentences and their embeddings
    (already computed), break between two consecutive sentences wherever
    they're LESS similar than most other consecutive pairs in the document
    -- i.e. a bigger topic jump than usual for this text.

    sentence_vecs is precomputed and passed in, rather than embedded here,
    so callers can batch every sentence across every document into a single
    API call (see rag.py) instead of one call per document -- this matters
    a lot under a 3 requests/minute rate limit.
    """
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return [{"doc": doc_name, "text": text.strip(), "strategy": "semantic"}] if text.strip() else []

    vecs = sentence_vecs
    norm = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    sims = np.sum(norm[:-1] * norm[1:], axis=1)  # consecutive-sentence cosine similarities

    threshold = np.percentile(sims, threshold_percentile)
    breakpoints = set(np.where(sims < threshold)[0] + 1)  # break BEFORE sentence i+1

    chunks, current = [], sentences[0]
    for i in range(1, len(sentences)):
        if i in breakpoints:
            chunks.append({"doc": doc_name, "text": current, "strategy": "semantic"})
            current = sentences[i]
        else:
            current = f"{current} {sentences[i]}"
    chunks.append({"doc": doc_name, "text": current, "strategy": "semantic"})
    return chunks


NON_SEMANTIC_STRATEGIES = {
    "fixed": chunk_fixed,
    "sentence": chunk_sentence,
    "recursive": chunk_recursive,
}
