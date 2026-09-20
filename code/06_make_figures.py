#!/usr/bin/env python3
"""Make small, readable figures used by the README and tutorial."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "word2vec_skipgram_100d.txt"
GRAPH = ROOT / "graph" / "concept_graph.gexf"
OUT = ROOT / "figures"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    wv = KeyedVectors.load_word2vec_format(str(MODEL), binary=False)
    words = [w for w in ["energy", "poverty", "digital", "carbon", "photovoltaic", "power", "employment", "insurance", "risk", "policy", "development", "technology", "sustainability", "model", "prediction", "investment"] if w in wv]
    coords = PCA(n_components=2, random_state=42).fit_transform(np.vstack([wv[w] for w in words]))
    plt.figure(figsize=(8, 6)); plt.scatter(coords[:,0], coords[:,1], s=30, color="#2563eb")
    for (x, y), word in zip(coords, words): plt.annotate(word, (x, y), fontsize=9, xytext=(4, 4), textcoords="offset points")
    plt.title("PCA projection of selected word vectors"); plt.xlabel("PC1"); plt.ylabel("PC2"); plt.tight_layout(); plt.savefig(OUT / "word_vectors_pca.png", dpi=180); plt.close()

    graph = nx.read_gexf(GRAPH)
    graph = nx.Graph(graph)
    # Display the strongest 25 edges to keep labels legible.
    ranked = sorted(graph.edges(data=True), key=lambda e: float(e[2].get("weight", 0)), reverse=True)[:25]
    view = nx.Graph(); view.add_edges_from((u, v, d) for u, v, d in ranked)
    for n in graph.nodes: view.add_node(n)
    degrees = dict(view.degree()); positions = nx.spring_layout(view, seed=42, k=0.8)
    plt.figure(figsize=(11, 8)); nx.draw_networkx_edges(view, positions, alpha=0.28, width=0.8, edge_color="#94a3b8")
    sizes = [70 + 35 * degrees.get(n, 0) for n in view.nodes]
    nx.draw_networkx_nodes(view, positions, node_size=sizes, node_color="#0f766e", alpha=0.88)
    labels = {n:n for n in view.nodes if degrees.get(n,0) >= 2}
    nx.draw_networkx_labels(view, positions, labels=labels, font_size=8)
    plt.title("Concept graph: strongest 25 observed edges"); plt.axis("off"); plt.tight_layout(); plt.savefig(OUT / "concept_graph_top25.png", dpi=180); plt.close()
    print(json.dumps({"pca_words": len(words), "graph_nodes": graph.number_of_nodes(), "graph_edges": graph.number_of_edges()}))


if __name__ == "__main__":
    main()
