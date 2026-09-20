#!/usr/bin/env python3
"""Build an attributed article–concept knowledge graph."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import networkx as nx
from gensim.models import KeyedVectors

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "corpus_sentences.jsonl"
MODEL = ROOT / "models" / "word2vec_skipgram_100d.txt"
OUT = ROOT / "graph"

# A small, explicit vocabulary makes the classroom graph inspectable. Students
# can replace it with keyword extraction in the extension exercise.
CONCEPTS = {
    "energy", "poverty", "development", "digital", "carbon", "emissions", "economy",
    "renewable", "coal", "consumption", "photovoltaic", "power", "generation", "model",
    "prediction", "insurance", "futures", "agricultural", "farmers", "risk", "utility",
    "coverage", "price", "employment", "prosperity", "index", "industry", "service",
    "pca", "stacking", "modernization", "sustainability", "coordination", "regional",
    "policy", "investment", "climate", "data", "machine", "learning", "lstm", "shap",
    "crra", "solar", "electricity", "income", "volatility", "premium", "guarantee",
    "target", "innovation", "governance", "business", "technology", "optimization",
    "forecasting", "regression", "panel", "spatial", "markov", "demand", "supply",
    "emission", "household",
}


def load_rows() -> list[dict]:
    with DATA.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def add_source_attr(data: dict, sentence_ids: list[int]) -> dict:
    # GEXF attributes are scalar; CSV retains the same IDs in a readable form.
    data["source_sentence_ids"] = ",".join(map(str, sorted(sentence_ids)))
    return data


def main() -> None:
    OUT.mkdir(exist_ok=True)
    rows = load_rows()
    articles = {}
    mention_ids: dict[tuple[str, str], list[int]] = defaultdict(list)
    cooccur_counts: Counter[tuple[str, str]] = Counter()
    cooccur_ids: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in rows:
        doc = row["doc_id"]; articles[doc] = row
        present = sorted(set(row["tokens"]) & CONCEPTS)
        for term in present:
            mention_ids[(doc, term)].append(int(row["sentence_id"]))
        for left, right in combinations(present, 2):
            cooccur_counts[(left, right)] += 1
            cooccur_ids[(left, right)].append(doc + ":" + str(row["sentence_id"]))

    full = nx.MultiDiGraph()
    for doc, row in sorted(articles.items()):
        full.add_node(f"article:{doc}", node_type="article", doc_id=doc, title=row["book_or_article"], doi=row["doi"], year=int(row["year"]))
    for term in sorted(CONCEPTS):
        full.add_node(f"term:{term}", node_type="concept", term=term)
    for (doc, term), ids in sorted(mention_ids.items()):
        full.add_edge(f"article:{doc}", f"term:{term}", relation="mentions", weight=len(ids), source_sentence_ids=",".join(map(str, sorted(ids))))
    for (left, right), count in sorted(cooccur_counts.items()):
        if count >= 2:
            full.add_edge(f"term:{left}", f"term:{right}", relation="co_occurs", weight=int(count), source_sentence_ids=",".join(map(str, sorted(cooccur_ids[(left, right)]))))

    semantic: dict[tuple[str, str], float] = {}
    if MODEL.exists():
        wv = KeyedVectors.load_word2vec_format(str(MODEL), binary=False)
        for term in sorted(CONCEPTS):
            if term not in wv:
                continue
            for neighbour, score in wv.most_similar(term, topn=5):
                if neighbour in CONCEPTS and neighbour != term and float(score) >= 0.75:
                    pair = tuple(sorted((term, neighbour)))
                    semantic[pair] = max(semantic.get(pair, 0.0), float(score))
    for (left, right), score in sorted(semantic.items()):
        full.add_edge(f"term:{left}", f"term:{right}", relation="semantic_similar", weight=round(score, 8), source_sentence_ids="")

    concept = nx.Graph()
    for term in sorted(CONCEPTS):
        concept.add_node(term, node_type="concept")
    for u, v, data in full.edges(data=True):
        if data.get("relation") in {"co_occurs", "semantic_similar"}:
            left, right = u.removeprefix("term:"), v.removeprefix("term:")
            old = concept.get_edge_data(left, right)
            if old is None or data["weight"] > old["weight"]:
                concept.add_edge(left, right, relation=data["relation"], weight=float(data["weight"]))

    nx.write_gexf(full, OUT / "knowledge_graph.gexf")
    nx.write_gexf(concept, OUT / "concept_graph.gexf")
    with (OUT / "nodes.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(["node_id", "node_type", "label", "title", "doi", "year"])
        for node, data in full.nodes(data=True):
            writer.writerow([node, data.get("node_type", ""), data.get("term", data.get("doc_id", "")), data.get("title", ""), data.get("doi", ""), data.get("year", "")])
    with (OUT / "edges.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(["source", "target", "relation", "weight", "source_sentence_ids"])
        for u, v, data in full.edges(data=True):
            writer.writerow([u, v, data.get("relation", ""), data.get("weight", ""), data.get("source_sentence_ids", "")])
    stats = {
        "article_nodes": sum(1 for _, d in full.nodes(data=True) if d.get("node_type") == "article"),
        "concept_nodes": len(CONCEPTS), "total_nodes": full.number_of_nodes(),
        "total_edges": full.number_of_edges(),
        "mention_edges": sum(1 for _, _, d in full.edges(data=True) if d.get("relation") == "mentions"),
        "cooccurrence_edges": sum(1 for _, _, d in full.edges(data=True) if d.get("relation") == "co_occurs"),
        "semantic_edges": sum(1 for _, _, d in full.edges(data=True) if d.get("relation") == "semantic_similar"),
        "cooccurrence_min_count": 2, "semantic_threshold": 0.75,
    }
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    (OUT / "concept_vocab.txt").write_text("\n".join(sorted(CONCEPTS)) + "\n", encoding="utf-8")
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
