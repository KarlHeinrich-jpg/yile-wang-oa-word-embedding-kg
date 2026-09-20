#!/usr/bin/env python3
"""Document vectors, TF–IDF comparison, and sentence-level semantic search."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "corpus_sentences.jsonl"
MODEL = ROOT / "models" / "word2vec_skipgram_100d.txt"
OUT = ROOT / "results"


def mean_vector(tokens, wv):
    vectors = [wv[t] for t in tokens if t in wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(wv.vector_size)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    rows = [json.loads(line) for line in DATA.open(encoding="utf-8")]
    wv = KeyedVectors.load_word2vec_format(str(MODEL), binary=False)
    by_doc = defaultdict(list)
    for row in rows: by_doc[row["doc_id"]].append(row)
    docs = sorted(by_doc)
    doc_texts = [" ".join(t for r in by_doc[d] for t in r["tokens"]) for d in docs]
    emb = np.vstack([mean_vector(text.split(), wv) for text in doc_texts])
    emb_sim = cosine_similarity(emb)
    tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w[\w'-]+\b", min_df=1)
    tfidf_sim = cosine_similarity(tfidf.fit_transform(doc_texts))
    with (OUT / "document_similarity.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(["doc_a", "doc_b", "word2vec_mean_cosine", "tfidf_cosine"])
        for i, a in enumerate(docs):
            for j in range(i + 1, len(docs)):
                writer.writerow([a, docs[j], f"{emb_sim[i,j]:.8f}", f"{tfidf_sim[i,j]:.8f}"])

    queries = ["energy poverty", "digital carbon", "photovoltaic power", "employment prosperity", "insurance futures"]
    sentence_vectors = np.vstack([mean_vector(r["tokens"], wv) for r in rows])
    with (OUT / "semantic_search.jsonl").open("w", encoding="utf-8") as f:
        for query in queries:
            qvec = mean_vector(query.split(), wv).reshape(1, -1)
            scores = cosine_similarity(qvec, sentence_vectors)[0]
            top = np.argsort(-scores)[:5]
            result = {"query": query, "hits": [{"doc_id": rows[i]["doc_id"], "sentence_id": rows[i]["sentence_id"], "score": round(float(scores[i]), 8), "text": rows[i]["text"]} for i in top]}
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    summary = {"documents": len(docs), "sentences": len(rows), "queries": queries, "embedding": "unweighted mean of Word2Vec vectors", "baseline": "TF-IDF cosine"}
    (OUT / "downstream_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
