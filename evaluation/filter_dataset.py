"""
Filter MetaMathQA to remove questions that overlap with the GSM8K test set
using n-gram matching. Any example whose query or original_question shares an
n-gram with a GSM8K test question is discarded.

This script was written in response to instructor guidance confirming that
n-gram deduplication against the GSM8K test set is sufficient for compliance.

Usage:
    python evaluation/filter_dataset.py
    python evaluation/filter_dataset.py --ngram_size 8 --output evaluation/metamath_filtered.jsonl

Output:
    JSONL file where each line is a MetaMathQA example (query, response, type,
    original_question) confirmed to have no n-gram overlap with the GSM8K test set.
"""

import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def get_ngrams(tokens: list[str], n: int) -> set[tuple]:
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def build_test_ngrams(ngram_size: int) -> set[tuple]:
    print("Loading GSM8K test set...")
    test = load_dataset("openai/gsm8k", "main", split="test")
    ngrams: set[tuple] = set()
    for ex in test:
        ngrams |= get_ngrams(tokenize(ex["question"]), ngram_size)
    print(f"  {len(ngrams):,} unique {ngram_size}-grams from {len(test)} test questions")
    return ngrams


def is_contaminated(text: str, test_ngrams: set[tuple], ngram_size: int) -> bool:
    return bool(get_ngrams(tokenize(text), ngram_size) & test_ngrams)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ngram_size", type=int, default=8)
    parser.add_argument("--output", type=str, default="evaluation/metamath_filtered.jsonl")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    test_ngrams = build_test_ngrams(args.ngram_size)

    print("Loading MetaMathQA (train split, streaming)...")
    dataset = load_dataset("meta-math/MetaMathQA", split="train", streaming=True)

    kept = removed = 0
    with open(output_path, "w") as f:
        for ex in dataset:
            # Check both the (possibly rephrased) query and the original source question
            contaminated = is_contaminated(ex.get("query", ""), test_ngrams, args.ngram_size) or \
                           is_contaminated(ex.get("original_question", ""), test_ngrams, args.ngram_size)
            if contaminated:
                removed += 1
            else:
                json.dump(ex, f)
                f.write("\n")
                kept += 1

            if (kept + removed) % 10_000 == 0:
                print(f"  processed {kept + removed:,}  kept {kept:,}  removed {removed:,}")

    total = kept + removed
    print(f"\nDone: {kept:,}/{total:,} kept, {removed:,} removed ({removed / total * 100:.1f}%)")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
