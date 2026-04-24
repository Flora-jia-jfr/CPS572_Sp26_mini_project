"""
Narrow grid search over the three hyperparameters most likely to improve
on the current best checkpoint:
    IFEval: 61.8%  GSM8K: 63.3%  HumanEval: 47.6%

Strategy:
  - Vary lr, (num_steps, patience) pairs, and gsm8k stage-1 weight
  - Fix everything else at the values that produced the best checkpoint
  - Directly monkey-patch train_and_publish globals instead of subprocess,
    so there's no process overhead and logs are all inline

Usage:
    python grid_search_narrow.py
    python grid_search_narrow.py --dry_run
    python grid_search_narrow.py --resume
    python grid_search_narrow.py --ids "lr4e-05_s400p5_gsm0.65"  # run one specific config

Results are saved incrementally to grid_search_narrow_results.json.
Each completed run appends immediately so a crash loses nothing.
"""

import argparse
import copy
import itertools
import json
import os
import random
import sys
import time
import types as pytypes
from datetime import datetime

# ── Grid definition ────────────────────────────────────────────────────────────
#
# We treat (num_steps, patience) as a PAIR because they interact: more steps
# only helps if patience is high enough to not early-stop before the hard stage.
#
# gsm8k_s1 controls the stage-1 GSM8K weight. Increasing it further may push
# GSM8K higher but will steal from HumanEval in stage 1. The tradeoff is the
# key unknown.
#
# lr: 4e-5 should preserve IFEval; 7e-5 may push HumanEval; 5e-5 is our
# current working value and the control.

GRID = {
    # (num_steps, patience) pairs — treated jointly
    "steps_patience": [
        (200, 5),   # ← current best, control
        (250, 6),   # let stage 2 breathe more
        (200, 3),   # exit stages faster, less per-stage overfitting
    ],
    "lr": [1e-5, 5e-5, 1e-4],          # 5e-5 is current best
    "gsm8k_s1": [0.55, 0.65, 0.75],    # 0.65 is current best
}
# Total: 3 × 3 × 3 = 27 configs

# ── Fixed hyperparameters (from best checkpoint) ───────────────────────────────
FIXED = {
    "batch_size":       16,
    "rank":             64,
    "val_every":        10,
    "val_batch_size":   64,
    "early_stopping":   True,
    "total_samples":    3000,
    # data_weights: [stage0, stage1, stage2] proportions of total steps
    "data_weights":     [0.25, 0.55, 0.20],
}

# ── Scoring ────────────────────────────────────────────────────────────────────
# Baselines from the assignment spec
BASELINES = {
    "ifeval":     0.45,
    "gsm8k":      0.50,
    "humaneval":  0.30,
}

# Weights reflect where we have the most room to improve
TASK_WEIGHTS = {
    "ifeval":    0.30,
    "gsm8k":     0.40,   # most room; also highest assignment weight
    "humaneval": 0.30,
}

def composite_score(metrics: dict) -> float:
    """
    Weighted score normalised against baselines.
    1.0 = exactly meeting all baselines; >1.0 = exceeding them.
    Returns 0.0 if no valid metrics found.
    """
    scores = {
        "ifeval":    metrics.get("ifeval_acc"),
        "gsm8k":     metrics.get("gsm8k_acc"),
        "humaneval": metrics.get("humaneval_acc"),
    }
    total, w_sum = 0.0, 0.0
    for task, val in scores.items():
        if val is not None:
            total += TASK_WEIGHTS[task] * (val / BASELINES[task])
            w_sum += TASK_WEIGHTS[task]
    return total / w_sum if w_sum > 0 else 0.0

# ── Config helpers ─────────────────────────────────────────────────────────────

def make_configs() -> list[dict]:
    keys = list(GRID.keys())
    combos = list(itertools.product(*[GRID[k] for k in keys]))
    configs = []
    for combo in combos:
        cfg = dict(zip(keys, combo))
        # Unpack the pair into separate fields for clarity
        cfg["num_steps"], cfg["patience"] = cfg.pop("steps_patience")
        configs.append(cfg)
    return configs

def config_id(cfg: dict) -> str:
    return (
        f"lr{cfg['lr']:.0e}"
        f"_s{cfg['num_steps']}p{cfg['patience']}"
        f"_gsm{cfg['gsm8k_s1']}"
    ).replace("e-0", "e-")

def build_stage_weights(gsm8k_s1: float) -> list[dict]:
    """
    Build stage_weights with gsm8k_s1 as the stage-1 GSM8K proportion.
    The remainder in stage 1 is split: 10% medium IFEval, 10% hard IFEval,
    rest goes to HumanEval to preserve it.

    Stage 0 and Stage 2 are fixed at the values that produced the best checkpoint.
    """
    # Stage 1: gsm8k_s1 to GSM8K, split the rest
    s1_remainder = 1.0 - gsm8k_s1
    # Fixed minor allocations in stage 1
    s1_if_medium = 0.10
    s1_if_hard   = 0.10
    s1_humaneval = max(0.0, round(s1_remainder - s1_if_medium - s1_if_hard, 4))

    return [
        # Stage 0 — warmup (fixed)
        {
            "if-eval easy":   0.35,
            "if-eval medium": 0.10,
            "if-eval hard":   0.00,
            "gsm8k":          0.40,
            "humaneval":      0.15,
        },
        # Stage 1 — math-dominant (gsm8k_s1 varies)
        {
            "if-eval easy":   0.00,
            "if-eval medium": s1_if_medium,
            "if-eval hard":   s1_if_hard,
            "gsm8k":          gsm8k_s1,
            "humaneval":      s1_humaneval,
        },
        # Stage 2 — code + hard IFEval (fixed)
        {
            "if-eval easy":   0.00,
            "if-eval medium": 0.00,
            "if-eval hard":   0.30,
            "gsm8k":          0.20,
            "humaneval":      0.50,
        },
    ]

# ── Patching train_and_publish ─────────────────────────────────────────────────

def run_training(cfg: dict, checkpoint_name: str) -> str | None:
    """
    Import train_and_publish and run it with patched globals.
    Returns the checkpoint path string, or None on failure.

    We patch the module-level constants and rebuild the argparse Namespace
    so we don't have to touch the file at all.
    """
    import importlib

    # Import (or reload to get a fresh module state)
    if "evaluation.train_and_publish" in sys.modules:
        tap = importlib.reload(sys.modules["evaluation.train_and_publish"])
    else:
        import evaluation.train_and_publish as tap
        sys.modules["evaluation.train_and_publish"] = tap

    # Patch the data/stage weights globals the script reads inside main()
    # We do this by temporarily replacing the module-level names if they exist,
    # OR by injecting them so main() picks them up via a closure trick below.
    #
    # Since main() defines data_weights / stage_weights as local variables,
    # the cleanest approach is to monkeypatch argparse.ArgumentParser so that
    # parse_args() returns our custom Namespace, and pass weight overrides
    # via a thread-local or module attribute.

    stage_weights = build_stage_weights(cfg["gsm8k_s1"])
    data_weights  = FIXED["data_weights"]

    # Inject overrides as module attributes; main() will check for these
    tap._GS_DATA_WEIGHTS  = data_weights
    tap._GS_STAGE_WEIGHTS = stage_weights

    # Build the args Namespace that main() would normally get from argparse
    fake_args = pytypes.SimpleNamespace(
        num_steps       = cfg["num_steps"],
        batch_size      = FIXED["batch_size"],
        lr              = cfg["lr"],
        rank            = FIXED["rank"],
        checkpoint_name = checkpoint_name,
        no_publish      = True,                 # never publish during grid search
        val_every       = FIXED["val_every"],
        val_batch_size  = FIXED["val_batch_size"],
        early_stopping  = FIXED["early_stopping"],
        patience        = cfg["patience"],
    )

    # Patch argparse so parse_args() inside main() returns our namespace
    import argparse as _argparse
    original_parse = _argparse.ArgumentParser.parse_args

    def patched_parse(self, args=None, namespace=None):
        return fake_args

    _argparse.ArgumentParser.parse_args = patched_parse

    # Also patch the data_weights / stage_weights local variables.
    # Since they're assigned inside main(), we wrap main() to intercept
    # the relevant assignments by providing them as module-level fallbacks
    # that main() checks (requires the one-line patch below in train_and_publish).
    #
    # ── ONE-TIME PATCH NEEDED in train_and_publish.py ──────────────────────────
    # In main(), replace the hardcoded lines:
    #
    #   data_weights = [0.25, 0.55, 0.2]
    #   stage_weights = [ ... ]
    #
    # with:
    #
    #   import sys
    #   _tap = sys.modules[__name__]
    #   data_weights  = getattr(_tap, "_GS_DATA_WEIGHTS",  [0.25, 0.55, 0.2])
    #   stage_weights = getattr(_tap, "_GS_STAGE_WEIGHTS", [ <original list> ])
    #
    # This is the ONLY change needed to train_and_publish.py.
    # Everything else (datasets, training loop, saving) runs unmodified.
    # ───────────────────────────────────────────────────────────────────────────

    checkpoint_path = None
    try:
        tap.main()
        # After main() runs, read the saved checkpoint_info.json
        info_path = os.path.join(tap.EVAL_DIR, "checkpoint_info.json")
        if os.path.exists(info_path):
            with open(info_path) as f:
                info = json.load(f)
            checkpoint_path = info.get("checkpoint_path")
    finally:
        # Restore argparse
        _argparse.ArgumentParser.parse_args = original_parse
        # Clean up injected attributes
        for attr in ("_GS_DATA_WEIGHTS", "_GS_STAGE_WEIGHTS"):
            if hasattr(tap, attr):
                delattr(tap, attr)

    return checkpoint_path


def run_eval(checkpoint_path: str, base_model: str, limit: int | None) -> dict:
    """
    Run eval_all on a checkpoint and return a flat metrics dict.
    Returns {} on failure.
    """
    import asyncio
    from evaluation.eval_all import run_core

    renderer_name = None  # auto-detect
    try:
        metrics, task_results = asyncio.run(run_core(
            base_model      = base_model,
            checkpoint_path = checkpoint_path,
            renderer_name   = renderer_name,
            temperature     = 0.0,
            top_p           = 1.0,
            limit           = limit,
            log_dir         = None,
            verbose         = False,
        ))
    except Exception as e:
        print(f"  [EVAL ERROR] {e}")
        return {}

    # Flatten to the three numbers we care about
    flat = {}
    # IFEval: average strict + loose prompt accuracy
    ifeval_strict = metrics.get("ifeval/prompt_level_strict_acc")
    ifeval_loose  = metrics.get("ifeval/prompt_level_loose_acc")
    if ifeval_strict is not None and ifeval_loose is not None:
        flat["ifeval_acc"] = (ifeval_strict + ifeval_loose) / 2
    elif ifeval_strict is not None:
        flat["ifeval_acc"] = ifeval_strict

    gsm = metrics.get("gsm8k/exact_match") or metrics.get("gsm8k/accuracy")
    if gsm is not None:
        flat["gsm8k_acc"] = gsm

    he = metrics.get("humaneval/pass@1") or metrics.get("humaneval/accuracy")
    if he is not None:
        flat["humaneval_acc"] = he

    flat["_raw"] = metrics   # keep full metrics for debugging
    return flat

# ── Results persistence ────────────────────────────────────────────────────────

def load_results(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"runs": [], "best": None}

def save_result(path: str, result: dict):
    data = load_results(path)
    ids = [r["id"] for r in data["runs"]]
    if result["id"] in ids:
        data["runs"][ids.index(result["id"])] = result
    else:
        data["runs"].append(result)
    # Update best
    scored = [r for r in data["runs"] if r.get("score") is not None]
    if scored:
        data["best"] = max(scored, key=lambda r: r["score"])
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    best = data.get("best")
    print(f"  [SAVED] best so far: {best['id'] if best else 'none'}"
          f"  score={best['score']:.4f}" if best and best.get('score') else "")

# ── Summary ────────────────────────────────────────────────────────────────────

def print_summary(results_path: str):
    data = load_results(results_path)
    runs = sorted(
        [r for r in data["runs"] if r.get("score") is not None],
        key=lambda r: r["score"],
        reverse=True,
    )
    if not runs:
        print("No completed runs to summarise.")
        return

    # Also flag which individual tasks improved over baseline
    print(f"\n{'='*95}")
    print(f"{'GRID SEARCH RESULTS':^95}")
    print(f"{'='*95}")
    print(f"  Current best checkpoint scores: IFEval=61.8%  GSM8K=63.3%  HumanEval=47.6%")
    print(f"{'─'*95}")
    header = (f"{'#':<4} {'Config ID':<38} {'Score':>7}"
              f" {'IFEval':>8} {'GSM8K':>8} {'HumanEval':>10}"
              f" {'IF↑':>5} {'G↑':>5} {'H↑':>5}")
    print(header)
    print("─" * 95)

    PREV = {"ifeval_acc": 0.618, "gsm8k_acc": 0.633, "humaneval_acc": 0.476}

    for rank, r in enumerate(runs, 1):
        m  = r.get("metrics", {})
        ie = m.get("ifeval_acc",    float("nan"))
        gs = m.get("gsm8k_acc",     float("nan"))
        he = m.get("humaneval_acc", float("nan"))
        sc = r["score"]
        ie_delta = f"+{ie - PREV['ifeval_acc']:.3f}"    if ie == ie else "  n/a"
        gs_delta = f"+{gs - PREV['gsm8k_acc']:.3f}"     if gs == gs else "  n/a"
        he_delta = f"+{he - PREV['humaneval_acc']:.3f}"  if he == he else "  n/a"
        print(f"{rank:<4} {r['id']:<38} {sc:>7.4f}"
              f" {ie:>8.3f} {gs:>8.3f} {he:>10.3f}"
              f" {ie_delta:>5} {gs_delta:>5} {he_delta:>5}")

    print("=" * 95)
    print(f"\nBest config: {data['best']['id']}")
    print(f"  Config:  {data['best']['config']}")
    print(f"  Metrics: {data['best'].get('metrics', {})}")

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Narrow grid search over lr, steps/patience, gsm8k_s1")
    parser.add_argument("--base_model", default="meta-llama/Llama-3.1-8B",
                        help="Base model (use 3B for speed during search)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Eval samples per task. None=full eval. 100 gives a fast signal.")
    parser.add_argument("--results", default="grid_search_narrow_results.json")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print all configs without running")
    parser.add_argument("--resume", action="store_true",
                        help="Skip configs already recorded in results file")
    parser.add_argument("--ids", nargs="+", default=None,
                        help="Run only these config IDs (space-separated)")
    parser.add_argument("--summary", action="store_true",
                        help="Print summary of existing results and exit")
    args = parser.parse_args()

    if args.summary:
        print_summary(args.results)
        return

    configs = make_configs()
    print(f"Grid: {len(GRID['steps_patience'])} step/patience pairs"
          f" × {len(GRID['lr'])} lrs"
          f" × {len(GRID['gsm8k_s1'])} gsm8k_s1 values"
          f" = {len(configs)} configs total")

    # Filters
    existing_ids: set[str] = set()
    if args.resume:
        data = load_results(args.results)
        existing_ids = {r["id"] for r in data["runs"] if r.get("score") is not None}
        print(f"Resuming: {len(existing_ids)} already done.")

    if args.ids:
        configs = [c for c in configs if config_id(c) in args.ids]
        print(f"Filtered to {len(configs)} specific configs.")

    if args.dry_run:
        print("\nDRY RUN — configs that would run:\n")
        for i, cfg in enumerate(configs):
            cid = config_id(cfg)
            skip = " [SKIP - already done]" if cid in existing_ids else ""
            sw = build_stage_weights(cfg["gsm8k_s1"])
            print(f"  [{i+1:>2}] {cid}{skip}")
            print(f"        lr={cfg['lr']:.0e}  steps={cfg['num_steps']}  patience={cfg['patience']}")
            print(f"        stage1 weights: gsm8k={sw[1]['gsm8k']}  humaneval={sw[1]['humaneval']}")
        return

    print("\nNOTE: train_and_publish.py requires a one-time patch to read")
    print("  _GS_DATA_WEIGHTS / _GS_STAGE_WEIGHTS from the module namespace.")
    print("  See the docstring in run_training() for the exact two lines to add.\n")

    for i, cfg in enumerate(configs):
        cid = config_id(cfg)
        if cid in existing_ids:
            print(f"[{i+1}/{len(configs)}] SKIP {cid}")
            continue

        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(configs)}] {cid}")
        print(f"  lr={cfg['lr']:.0e}  steps={cfg['num_steps']}  patience={cfg['patience']}"
              f"  gsm8k_s1={cfg['gsm8k_s1']}")
        sw = build_stage_weights(cfg["gsm8k_s1"])
        print(f"  stage1: gsm8k={sw[1]['gsm8k']}  humaneval={sw[1]['humaneval']}"
              f"  if_medium={sw[1]['if-eval medium']}  if_hard={sw[1]['if-eval hard']}")
        print(f"{'='*70}")

        result = {
            "id":        cid,
            "config":    {k: cfg[k] for k in ("lr", "num_steps", "patience", "gsm8k_s1")},
            "timestamp": datetime.utcnow().isoformat(),
            "checkpoint_path": None,
            "metrics":   {},
            "score":     None,
            "error":     None,
        }

        # ── Train ──────────────────────────────────────────────────────────────
        t0 = time.time()
        try:
            ckpt = run_training(cfg, checkpoint_name=cid)
            result["checkpoint_path"] = ckpt
            result["train_time_s"] = round(time.time() - t0, 1)
            print(f"  [TRAIN] done in {result['train_time_s']}s  ckpt={ckpt}")
        except Exception as e:
            result["error"] = f"train: {e}"
            print(f"  [TRAIN ERROR] {e}")
            save_result(args.results, result)
            continue

        if not result["checkpoint_path"]:
            result["error"] = "no checkpoint path returned"
            save_result(args.results, result)
            continue

        # ── Eval ───────────────────────────────────────────────────────────────
        t1 = time.time()
        metrics = run_eval(result["checkpoint_path"], args.base_model, args.limit)
        result["eval_time_s"] = round(time.time() - t1, 1)
        result["metrics"] = {k: v for k, v in metrics.items() if k != "_raw"}
        result["metrics_raw"] = metrics.get("_raw", {})
        result["score"] = composite_score(metrics) if metrics else None

        print(f"  [EVAL]  IFEval={metrics.get('ifeval_acc', 'n/a'):.3f}"
              f"  GSM8K={metrics.get('gsm8k_acc', 'n/a'):.3f}"
              f"  HumanEval={metrics.get('humaneval_acc', 'n/a'):.3f}"
              f"  → score={result['score']:.4f}" if result["score"] else "")

        save_result(args.results, result)

    print_summary(args.results)


if __name__ == "__main__":
    main()