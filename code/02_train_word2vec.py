#!/usr/bin/env python3
"""Train a deterministic, CPU-friendly Skip-gram Word2Vec model."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from gensim.models import Word2Vec

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "processed" / "corpus_tokenized.txt"
MODEL_DIR = ROOT / "models"
RESULTS = ROOT / "results"


def stable_hash(value) -> int:
    raw = value if isinstance(value, bytes) else str(value).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "little")


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True); RESULTS.mkdir(exist_ok=True)
    model = Word2Vec(
        corpus_file=str(CORPUS), vector_size=100, window=5, min_count=2,
        workers=1, sg=1, negative=10, sample=1e-4, epochs=20,
        seed=42, hashfxn=stable_hash,
    )
    model.save(str(MODEL_DIR / "word2vec_skipgram_100d.model"))
    model.wv.save_word2vec_format(str(MODEL_DIR / "word2vec_skipgram_100d.txt"), binary=False)
    model.wv.save_word2vec_format(str(MODEL_DIR / "word2vec_skipgram_100d.bin"), binary=True)
    probes = [w for w in ["energy", "poverty", "digital", "carbon", "photovoltaic", "employment", "insurance"] if w in model.wv]
    with (RESULTS / "word2vec_neighbors.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(["query", "neighbor", "cosine"])
        for query in probes:
            for neighbor, score in model.wv.most_similar(query, topn=8):
                writer.writerow([query, neighbor, f"{score:.8f}"])
    summary = {
        "vocabulary": len(model.wv), "vector_size": model.vector_size,
        "corpus_count": model.corpus_count, "corpus_total_words": model.corpus_total_words,
        "window": 5, "min_count": 2, "sg": 1, "negative": 10,
        "sample": 1e-4, "epochs": 20, "workers": 1, "seed": 42,
        "gensim_version": __import__("gensim").__version__,
    }
    (RESULTS / "word2vec_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
