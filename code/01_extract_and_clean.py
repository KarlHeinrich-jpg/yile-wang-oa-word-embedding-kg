#!/usr/bin/env python3
"""Extract article text and write a sentence-level, tokenized corpus.

JATS XML is preferred because it preserves paragraph boundaries. PDF extraction
is a fallback for the three articles supplied only as PDFs. The rules are
intentionally readable for a classroom; they are not a publication-grade PDF
layout parser.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable

import fitz
from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
MANIFEST = RAW / "source_manifest.json"
TOKEN_RE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z0-9]+)*|\d+(?:\.\d+)?")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])")
ABBREVIATIONS = {"e.g.", "i.e.", "et al.", "Fig.", "Eq.", "No.", "Dr.", "vs."}


def normalise_space(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("\u00ad", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    text = normalise_space(text)
    if not text:
        return []
    # Protect a small set of common abbreviations before splitting.
    protected = {a: a.replace(".", "<DOT>") for a in ABBREVIATIONS}
    for old, new in protected.items():
        text = text.replace(old, new)
    pieces = SENTENCE_RE.split(text)
    restored = []
    for piece in pieces:
        for old, new in protected.items():
            piece = piece.replace(new, old)
        piece = normalise_space(piece)
        if len(piece) >= 20 and re.search(r"[A-Za-z]", piece):
            restored.append(piece)
    return restored


def tokens(sentence: str) -> list[str]:
    return [x.lower() for x in TOKEN_RE.findall(sentence)]


def extract_xml(path: Path) -> str:
    parser = etree.XMLParser(recover=True, huge_tree=True)
    tree = etree.parse(str(path), parser)
    chunks: list[str] = []
    for node in tree.xpath("//article-title | //abstract//p | //body//p | //body//title"):
        text = normalise_space(" ".join(node.itertext()))
        if text and text not in chunks:
            chunks.append(text)
    return "\n".join(chunks)


def extract_pdf(path: Path) -> str:
    pages: list[str] = []
    for page in fitz.open(str(path)):
        text = normalise_space(page.get_text("text"))
        if text:
            pages.append(text)
    text = "\n".join(pages)
    # Reference lists add bibliographic noise and should not train the example.
    text = re.split(r"\b(?:references|bibliography)\b", text, maxsplit=1, flags=re.I)[0]
    lines = [normalise_space(x) for x in text.splitlines()]
    counts = Counter(x for x in lines if 3 < len(x) < 140)
    repeated = {line for line, n in counts.items() if n >= 4}
    lines = [line for line in lines if line not in repeated and not re.fullmatch(r"\d+", line)]
    return "\n".join(lines)


def source_text(article: dict) -> tuple[str, str]:
    xml = RAW / f"{article['doc_id']}.xml"
    pdf = RAW / f"{article['doc_id']}.pdf"
    if xml.exists():
        return extract_xml(xml), "JATS XML"
    if pdf.exists():
        return extract_pdf(pdf), "PDF"
    # The committed processed corpus makes a clone usable before a raw download.
    return "", "committed processed corpus"


def existing_processed() -> bool:
    return (OUT / "corpus_sentences.jsonl").exists() and (OUT / "corpus_tokenized.txt").exists()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-extract after raw XML/PDF files are downloaded")
    args = parser.parse_args()
    if existing_processed() and not args.force:
        print("processed corpus exists; reusing it (use --force after downloading raw files)")
        return
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    raw_blocks: list[str] = []
    for article in manifest["articles"]:
        text, source_kind = source_text(article)
        if not text:
            raise SystemExit(f"no raw source for {article['doc_id']}; run code/00_download_sources.py first")
        article_rows = []
        seen: set[str] = set()
        for sentence in split_sentences(text):
            tok = tokens(sentence)
            if len(tok) < 4:
                continue
            key = " ".join(tok)
            if key in seen:
                continue
            seen.add(key)
            row = {
                "doc_id": article["doc_id"],
                "book_or_article": article["title"],
                "doi": article["doi"],
                "year": article["year"],
                "authors": article["authors"],
                "license": article["license"],
                "license_url": article["license_url"],
                "publisher_url": article["publisher_url"],
                "sentence_id": len(article_rows),
                "text": sentence,
                "tokens": tok,
                "source_kind": source_kind,
            }
            article_rows.append(row)
        all_rows.extend(article_rows)
        raw_blocks.append("\n".join(r["text"] for r in article_rows))
        stem = article["doc_id"]
        (OUT / "by_article" ).mkdir(exist_ok=True)
        (OUT / "by_article" / f"{stem}.txt").write_text("\n".join(r["text"] for r in article_rows) + "\n", encoding="utf-8")
        (OUT / "by_article" / f"{stem}.tokenized.txt").write_text("\n".join(" ".join(r["tokens"]) for r in article_rows) + "\n", encoding="utf-8")
    with (OUT / "corpus_sentences.jsonl").open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (OUT / "corpus_sentences.txt").write_text("\n".join(r["text"] for r in all_rows) + "\n", encoding="utf-8")
    (OUT / "corpus_tokenized.txt").write_text("\n".join(" ".join(r["tokens"]) for r in all_rows) + "\n", encoding="utf-8")
    (OUT / "corpus_raw_clean.txt").write_text("\n\n".join(raw_blocks) + "\n", encoding="utf-8")
    freq = Counter(t for r in all_rows for t in r["tokens"])
    with (OUT / "token_frequencies.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(["token", "count"])
        writer.writerows(freq.most_common())
    summary = {"articles": len(manifest["articles"]), "sentences": len(all_rows), "tokens": sum(freq.values()), "unique_tokens": len(freq), "tokenizer": TOKEN_RE.pattern}
    (OUT / "processing_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
