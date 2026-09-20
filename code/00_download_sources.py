#!/usr/bin/env python3
"""Download the OA source files listed in data/raw/source_manifest.json.

The repository already contains processed text, so downloading is optional for a
first run. Raw files are deliberately ignored by .gitignore. Keep the DOI and
license metadata with any copy you retain.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MANIFEST = RAW / "source_manifest.json"


def download(url: str, path: Path) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "yile-wang-word-embedding-teaching/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=["all", "xml", "pdf"], default="all")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    records = []
    for article in manifest["articles"]:
        targets = []
        if args.format in {"all", "xml"} and article.get("xml_url"):
            targets.append((article["xml_url"], RAW / f"{article['doc_id']}.xml"))
        if args.format in {"all", "pdf"} and article.get("pdf_url"):
            targets.append((article["pdf_url"], RAW / f"{article['doc_id']}.pdf"))
        for url, path in targets:
            record = {"doc_id": article["doc_id"], "url": url, "path": path.name}
            try:
                record["sha256"] = download(url, path)
                record["status"] = "ok"
                print(f"downloaded {path.name} ({path.stat().st_size:,} bytes)")
            except Exception as exc:
                record["status"] = "error"
                record["error"] = str(exc)
                print(f"could not download {url}: {exc}", file=sys.stderr)
            records.append(record)
    (RAW / "download_manifest.json").write_text(
        json.dumps({"downloaded_on": str(date.today()), "files": records}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if any(r["status"] == "error" for r in records):
        raise SystemExit("one or more downloads failed; inspect data/raw/download_manifest.json")


if __name__ == "__main__":
    main()
