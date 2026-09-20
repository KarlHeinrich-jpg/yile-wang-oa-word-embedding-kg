#!/usr/bin/env python3
"""Run the complete classroom pipeline from the repository root."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(path: str, *args: str) -> None:
    command = [sys.executable, str(ROOT / "code" / path), *args]
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True, env={**os.environ, "PYTHONHASHSEED": "0"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-extract", action="store_true", help="reuse committed or existing processed corpus")
    args = parser.parse_args()
    if not args.skip_extract:
        run("01_extract_and_clean.py")
    run("02_train_word2vec.py")
    run("03_build_knowledge_graph.py")
    run("04_link_prediction.py")
    run("05_downstream_recommendation.py")
    run("06_make_figures.py")
    print("pipeline complete")


if __name__ == "__main__":
    main()
