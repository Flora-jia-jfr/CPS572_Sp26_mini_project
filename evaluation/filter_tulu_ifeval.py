"""
Filter allenai/tulu-3-sft-mixture to keep only the IFEval-specific source:
  ai2-adapt-dev/personahub_ifdata_manual_seed_v3_29980

This subset (~30K examples) was specifically constructed to target IFEval
constraint-following categories (formatting, length, keywords, postscript, etc.)

Usage:
    python evaluation/filter_tulu_ifeval.py
    python evaluation/filter_tulu_ifeval.py --output evaluation/tulu_ifeval_filtered.jsonl
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset

IFEVAL_SOURCE = "ai2-adapt-dev/personahub_ifdata_manual_seed_v3_29980"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="evaluation/tulu_ifeval_filtered.jsonl")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading allenai/tulu-3-sft-mixture (streaming)...")
    print(f"Keeping only source: {IFEVAL_SOURCE}")
    ds = load_dataset("allenai/tulu-3-sft-mixture", split="train", streaming=True)

    kept = skipped = 0
    with open(output_path, "w") as f:
        for ex in ds:
            if ex.get("source") == IFEVAL_SOURCE:
                json.dump(ex, f)
                f.write("\n")
                kept += 1
                if kept % 5_000 == 0:
                    print(f"  kept {kept:,} so far...")
            else:
                skipped += 1

    print(f"\nDone: {kept:,} kept, {skipped:,} skipped")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
