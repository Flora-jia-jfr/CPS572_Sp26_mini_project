"""
Verify that MetaMathQA (raw and filtered) does not overlap with the GSM8K test set.

Checks both the raw HuggingFace dataset and the locally filtered JSONL (if present),
and reports overlap statistics for each.

Usage:
    python evaluation/verify_contamination.py
    python evaluation/verify_contamination.py --ngram_size 8 --show_examples
"""

import argparse
import json
import os
import re

from datasets import load_dataset

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
METAMATH_FILTERED = os.path.join(EVAL_DIR, "metamath_filtered.jsonl")


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def get_ngrams(tokens: list[str], n: int) -> set[tuple]:
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def build_test_ngrams(ngram_size: int) -> tuple[set[tuple], list[str]]:
    print("Loading GSM8K test set...")
    test = load_dataset("openai/gsm8k", "main", split="test")
    ngrams: set[tuple] = set()
    questions = []
    for ex in test:
        q = ex["question"]
        questions.append(q)
        ngrams |= get_ngrams(tokenize(q), ngram_size)
    print(f"  {len(test)} test questions → {len(ngrams):,} unique {ngram_size}-grams")
    return ngrams, questions


def check_overlap(
    examples: list[dict],
    test_ngrams: set[tuple],
    ngram_size: int,
    text_field: str,
    show_examples: bool,
) -> int:
    overlapping = 0
    for ex in examples:
        text = ex.get(text_field, "")
        if get_ngrams(tokenize(text), ngram_size) & test_ngrams:
            overlapping += 1
            if show_examples and overlapping <= 3:
                print(f"    OVERLAP: {text[:120]}...")
    return overlapping


def check_dataset(
    name: str,
    examples: list[dict],
    test_ngrams: set[tuple],
    ngram_size: int,
    show_examples: bool,
):
    total = len(examples)
    print(f"\n{'='*60}")
    print(f"Checking: {name}  ({total:,} examples)")
    print(f"{'='*60}")

    # Check query field
    overlap_query = check_overlap(examples, test_ngrams, ngram_size, "query", show_examples)
    # Check original_question field (present in MetaMathQA)
    overlap_orig = check_overlap(examples, test_ngrams, ngram_size, "original_question", show_examples)
    # Union: contaminated if either field overlaps
    contaminated = sum(
        1 for ex in examples
        if (get_ngrams(tokenize(ex.get("query", "")), ngram_size) & test_ngrams)
        or (get_ngrams(tokenize(ex.get("original_question", "")), ngram_size) & test_ngrams)
    )

    print(f"  query overlaps:             {overlap_query:,} / {total:,}  ({overlap_query/total*100:.2f}%)")
    print(f"  original_question overlaps: {overlap_orig:,} / {total:,}  ({overlap_orig/total*100:.2f}%)")
    print(f"  total contaminated:         {contaminated:,} / {total:,}  ({contaminated/total*100:.2f}%)")
    if contaminated == 0:
        print("  RESULT: CLEAN — no overlap with GSM8K test set")
    else:
        print("  RESULT: CONTAMINATED — run filter_dataset.py before training")
    return contaminated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ngram_size", type=int, default=8)
    parser.add_argument("--show_examples", action="store_true", help="Print up to 3 overlapping examples per dataset")
    parser.add_argument("--raw_limit", type=int, default=50_000, help="Max raw MetaMathQA examples to sample (default 50k for speed)")
    args = parser.parse_args()

    test_ngrams, _ = build_test_ngrams(args.ngram_size)

    # --- Check raw MetaMathQA (sampled for speed) ---
    print(f"\nLoading MetaMathQA raw (first {args.raw_limit:,} examples)...")
    raw_ds = load_dataset("meta-math/MetaMathQA", split="train", streaming=True)
    raw_examples = []
    for ex in raw_ds:
        raw_examples.append(ex)
        if len(raw_examples) >= args.raw_limit:
            break
    check_dataset(
        f"MetaMathQA raw (sample of {len(raw_examples):,})",
        raw_examples,
        test_ngrams,
        args.ngram_size,
        args.show_examples,
    )

    # --- Check filtered JSONL if present ---
    if os.path.exists(METAMATH_FILTERED):
        print(f"\nLoading filtered JSONL: {METAMATH_FILTERED}")
        with open(METAMATH_FILTERED) as f:
            filtered_examples = [json.loads(line) for line in f]
        check_dataset(
            "MetaMathQA filtered (full)",
            filtered_examples,
            test_ngrams,
            args.ngram_size,
            args.show_examples,
        )
    else:
        print(f"\nNo filtered file found at {METAMATH_FILTERED} — skipping filtered check.")
        print("Run: python evaluation/filter_dataset.py")


if __name__ == "__main__":
    main()
