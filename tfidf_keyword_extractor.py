"""Command-line utility for extracting keywords from Chinese accident reports.

This script walks through a directory, loads every `.txt` file, performs
tokenisation with Jieba, and computes TF-IDF scores to identify the most
important accident-cause keywords in each document. Optionally, a stop-word
file can be supplied to filter out common words.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, List, Sequence, Set

import jieba


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract accident-cause keywords from a directory of Chinese "
            "accident investigation reports stored as .txt files using Jieba "
            "tokenisation and TF-IDF scoring."
        )
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Path to the directory that contains .txt accident reports.",
    )
    parser.add_argument(
        "--stopwords",
        type=Path,
        default=None,
        help=(
            "Optional path to a UTF-8 encoded text file where each line is a "
            "stop word to be ignored during tokenisation."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of keywords to output for each document (default: 20).",
    )
    parser.add_argument(
        "--min-token-length",
        type=int,
        default=2,
        help=(
            "Filter out tokens shorter than this length after stripping "
            "whitespace (default: 2)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional output JSON file. When omitted, results are printed to "
            "stdout."
        ),
    )
    return parser.parse_args(argv)


def load_stopwords(stopword_path: Path | None) -> Set[str]:
    if stopword_path is None:
        return set()

    with stopword_path.open("r", encoding="utf-8") as f:
        stopwords = {line.strip() for line in f if line.strip()}
    return stopwords


def iter_txt_files(directory: Path) -> Iterable[Path]:
    for path in sorted(directory.rglob("*.txt")):
        if path.is_file():
            yield path


def tokenise(text: str, stopwords: Set[str], min_len: int) -> List[str]:
    tokens: List[str] = []
    for token in jieba.cut(text):
        token = token.strip()
        if not token:
            continue
        if len(token) < min_len:
            continue
        if token in stopwords:
            continue
        tokens.append(token)
    return tokens


def compute_tf_idf(
    documents: Sequence[Sequence[str]],
) -> tuple[list[Counter[str]], dict[str, float]]:
    tf_counters: list[Counter[str]] = []
    df_counter: Counter[str] = Counter()

    for tokens in documents:
        counter = Counter(tokens)
        tf_counters.append(counter)
        unique_terms = set(counter.keys())
        df_counter.update(unique_terms)

    doc_count = len(documents)
    idf: dict[str, float] = {}
    for term, df in df_counter.items():
        # Add-one smoothing on DF to avoid division by zero when df == doc_count.
        idf[term] = math.log((1 + doc_count) / (1 + df)) + 1
    return tf_counters, idf


def extract_keywords(
    tf_counter: Counter[str],
    idf: dict[str, float],
    top_k: int,
) -> list[tuple[str, float]]:
    scores = []
    total_terms = sum(tf_counter.values()) or 1
    for term, freq in tf_counter.items():
        tf = freq / total_terms
        score = tf * idf.get(term, 0.0)
        scores.append((term, score))
    scores.sort(key=lambda item: item[1], reverse=True)
    return scores[:top_k]


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)

    if not args.input_dir.exists() or not args.input_dir.is_dir():
        print(f"Input directory {args.input_dir} does not exist or is not a directory.", file=sys.stderr)
        return 1

    stopwords = load_stopwords(args.stopwords)

    documents: list[list[str]] = []
    file_paths: list[Path] = []

    for file_path in iter_txt_files(args.input_dir):
        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        tokens = tokenise(text, stopwords, args.min_token_length)
        if tokens:
            documents.append(tokens)
            file_paths.append(file_path)

    if not documents:
        print("No valid documents were found in the specified directory.", file=sys.stderr)
        return 1

    tf_counters, idf = compute_tf_idf(documents)

    results: dict[str, list[dict[str, float]]] = defaultdict(list)
    for file_path, tf_counter in zip(file_paths, tf_counters):
        keywords = extract_keywords(tf_counter, idf, args.top_k)
        results[str(file_path)].extend(
            [{"keyword": term, "score": round(score, 6)} for term, score in keywords]
        )

    if args.output:
        args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print(json.dumps(results, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
