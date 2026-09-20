#!/usr/bin/env python3
"""Evaluate graph link prediction without test-edge or label leakage.

The graph contains observed co-occurrence edges only for this experiment. A
60/20/20 split is made from true edges; negatives come exclusively from pairs
that are absent from the complete observed graph. Pair features are calculated
from the 60% training graph, so validation and test edges are absent when their
features are created.
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
EDGES = ROOT / "graph" / "edges.csv"
OUT = ROOT / "results"
FIG = ROOT / "figures"
SEED = 42


def canonical(u: str, v: str) -> tuple[str, str]:
    return tuple(sorted((u, v)))


def load_graph() -> nx.Graph:
    graph = nx.Graph()
    with EDGES.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["relation"] == "co_occurs":
                graph.add_edge(row["source"].removeprefix("term:"), row["target"].removeprefix("term:"))
    return graph


def feature_row(graph: nx.Graph, pair: tuple[str, str]) -> list[float]:
    u, v = pair
    common = set(nx.common_neighbors(graph, u, v))
    union = set(graph[u]) | set(graph[v])
    cn = float(len(common))
    jaccard = cn / len(union) if union else 0.0
    aa = sum(1.0 / np.log(graph.degree(w)) for w in common if graph.degree(w) > 1)
    pa = float(graph.degree(u) * graph.degree(v))
    return [cn, jaccard, aa, pa]


def sample_negatives(full: nx.Graph, n: int, forbidden: set[tuple[str, str]], rng: random.Random) -> list[tuple[str, str]]:
    candidates = sorted(canonical(u, v) for u, v in nx.non_edges(full) if canonical(u, v) not in forbidden)
    rng.shuffle(candidates)
    if len(candidates) < n:
        raise ValueError("not enough true non-edges for negative sampling")
    return candidates[:n]


def main() -> None:
    OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
    full = load_graph()
    all_pos = sorted(canonical(u, v) for u, v in full.edges())
    rng = random.Random(SEED); rng.shuffle(all_pos)
    n_test = max(1, int(round(len(all_pos) * 0.20)))
    n_valid = max(1, int(round(len(all_pos) * 0.20)))
    test_pos = all_pos[:n_test]; valid_pos = all_pos[n_test:n_test+n_valid]; train_pos = all_pos[n_test+n_valid:]
    used = set(all_pos)
    train_neg = sample_negatives(full, len(train_pos), used, rng); used.update(train_neg)
    valid_neg = sample_negatives(full, len(valid_pos), used, rng); used.update(valid_neg)
    test_neg = sample_negatives(full, len(test_pos), used, rng)
    base = nx.Graph(); base.add_nodes_from(full.nodes()); base.add_edges_from(train_pos)
    def matrix(pairs): return np.asarray([feature_row(base, p) for p in pairs], dtype=float)
    def leave_one_out(graph, pairs):
        features = []
        for pair in pairs:
            exists = graph.has_edge(*pair)
            if exists: graph.remove_edge(*pair)
            features.append(feature_row(graph, pair))
            if exists: graph.add_edge(*pair)
        return np.asarray(features)
    X_train = np.vstack([leave_one_out(base, train_pos), matrix(train_neg)])
    y_train = np.r_[np.ones(len(train_pos)), np.zeros(len(train_neg))]
    X_valid = np.vstack([matrix(valid_pos), matrix(valid_neg)])
    y_valid = np.r_[np.ones(len(valid_pos)), np.zeros(len(valid_neg))]
    X_test = np.vstack([matrix(test_pos), matrix(test_neg)])
    y_test = np.r_[np.ones(len(test_pos)), np.zeros(len(test_neg))]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED))
    clf.fit(X_train, y_train)
    valid_score = clf.predict_proba(X_valid)[:, 1]
    test_score = clf.predict_proba(X_test)[:, 1]
    metrics = {
        "protocol": "60/20/20 edge split; graph-only features computed from training positives",
        "seed": SEED, "full_cooccurrence_edges": len(all_pos),
        "train_positive": len(train_pos), "validation_positive": len(valid_pos), "test_positive": len(test_pos),
        "train_negative": len(train_neg), "validation_negative": len(valid_neg), "test_negative": len(test_neg),
        "roc_auc_validation": float(roc_auc_score(y_valid, valid_score)),
        "average_precision_validation": float(average_precision_score(y_valid, valid_score)),
        "roc_auc_test": float(roc_auc_score(y_test, test_score)),
        "average_precision_test": float(average_precision_score(y_test, test_score)),
        "split_checks": {
            "test_positive_absent_from_train_graph": all(not base.has_edge(*p) for p in test_pos),
            "test_negatives_absent_from_full_graph": all(not full.has_edge(*p) for p in test_neg),
            "train_valid_test_positive_disjoint": not (set(train_pos)&set(valid_pos) or set(train_pos)&set(test_pos) or set(valid_pos)&set(test_pos)),
            "negative_sets_disjoint": not (set(train_neg)&set(valid_neg) or set(train_neg)&set(test_neg) or set(valid_neg)&set(test_neg)),
        },
        "feature_names": ["common_neighbors", "jaccard", "adamic_adar", "preferential_attachment"],
    }
    if not all(metrics["split_checks"].values()):
        raise AssertionError(metrics["split_checks"])
    (OUT / "link_prediction_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    # Fit on all observed edges only for candidate ranking after evaluation.
    final_base = full.copy()
    final_used = set(all_pos)
    final_neg = sample_negatives(full, len(all_pos), final_used, random.Random(SEED + 1))
    final_clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED))
    final_clf.fit(np.vstack([leave_one_out(final_base, all_pos), np.asarray([feature_row(final_base, p) for p in final_neg])]), np.r_[np.ones(len(all_pos)), np.zeros(len(final_neg))])
    candidates = []
    for pair in nx.non_edges(full):
        pair = canonical(*pair)
        score = float(final_clf.predict_proba(np.asarray([feature_row(final_base, pair)]))[:, 1][0])
        candidates.append((score, pair[0], pair[1], *feature_row(final_base, pair)))
    candidates.sort(reverse=True)
    with (OUT / "predicted_links.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(["score", "source", "target", "common_neighbors", "jaccard", "adamic_adar", "preferential_attachment"])
        writer.writerows([[f"{x:.8f}" if isinstance(x, float) else x for x in row] for row in candidates[:100]])
    top = candidates[:15]
    labels = [f"{a} — {b}" for _, a, b, *_ in reversed(top)]
    values = [x[0] for x in reversed(top)]
    plt.figure(figsize=(8, 6)); plt.barh(labels, values, color="#3b82f6"); plt.xlabel("ranking score"); plt.title("Top candidate concept links"); plt.tight_layout(); plt.savefig(FIG / "predicted_links.png", dpi=180); plt.close()
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
