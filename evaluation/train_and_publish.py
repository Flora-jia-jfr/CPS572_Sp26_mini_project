"""
Train a model (minimal SFT), save checkpoint, and publish it.

NOTE: This is a TOY EXAMPLE that trains for a few steps on dummy data
to verify the full workflow end-to-end. You should replace the training
data and training logic with your own implementation.

TODO:
  - Replace DEMO_CONVERSATIONS with your task-specific training data
  - Tune hyperparameters (learning rate, batch size, number of steps, LoRA rank)
  - Add validation / early stopping as needed

Usage:
    python evaluation/train_and_publish.py
    python evaluation/train_and_publish.py --num_steps 20
    python evaluation/train_and_publish.py --no_publish   # skip publishing
"""

import argparse
import json
import os

import numpy as np
import tinker
from tinker import types
from tinker_cookbook import model_info, renderers
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer
from itertools import cycle
from datasets import interleave_datasets, load_dataset

MODEL = "meta-llama/Llama-3.2-3B"
# MODEL = "meta-llama/Llama-3.2-1B"    # Smaller, faster for development
# MODEL = "meta-llama/Llama-3.1-8B"    # Recommended for final submission

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))

def gsm8k_to_conversation(example):
    return [
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["answer"]},
    ]


def tulu_to_conversation(example):
    # The dataset already provides instruction-tuning messages.
    messages = example["messages"]
    cleaned = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role in {"user", "assistant", "system"} and isinstance(content, str) and content.strip():
            cleaned.append({"role": role, "content": content})
    return cleaned


def opencode_to_conversation(example):
    return [
        {"role": "user", "content": example["input"]},
        {"role": "assistant", "content": example["output"]},
    ]

def is_valid_conversation(convo):
    if not convo or len(convo) < 2:
        return False
    has_user = any(m.get("role") == "user" and m.get("content", "").strip() for m in convo)
    has_assistant = any(m.get("role") == "assistant" and m.get("content", "").strip() for m in convo)
    return has_user and has_assistant

def build_training_iterator(renderer, max_length):
    # Stream train splits only. No test/validation splits are used.
    gsm8k = load_dataset("openai/gsm8k", "main", split="train", streaming=True)
    gsm8k = gsm8k.map(lambda ex: {"conversation": gsm8k_to_conversation(ex)})

    tulu = load_dataset("allenai/tulu-3-sft-mixture", split="train", streaming=True)
    tulu = tulu.map(lambda ex: {"conversation": tulu_to_conversation(ex)})

    opencode = load_dataset("nvidia/OpenCodeInstruct", split="train", streaming=True)
    opencode = opencode.map(lambda ex: {"conversation": opencode_to_conversation(ex)})

    # Equal-probability interleave so the 5M-row code dataset does not dominate.
    mixed = interleave_datasets(
        [gsm8k, tulu, opencode],
        probabilities = [0.45, 0.30, 0.25],
        seed=42,
        stopping_strategy="all_exhausted",
    )

    for ex in mixed:
        convo = ex["conversation"]
        if not is_valid_conversation(convo):
            continue
        try:
            datum = conversation_to_datum(
                convo,
                renderer,
                max_length=max_length,
                train_on_what=renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES,
            )
            yield datum
        except Exception:
            # Skip malformed / overlong / incompatible examples
            continue
def main():
    parser = argparse.ArgumentParser(description="Train, save, and publish a checkpoint")
    parser.add_argument("--num_steps", type=int, default=1500, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--rank", type=int, default=64, help="LoRA rank")
    parser.add_argument("--max_length", type=int, default=1536, help="Max token length")
    parser.add_argument("--checkpoint_name", type=str, default="demo", help="Checkpoint name")
    parser.add_argument("--no_publish", action="store_true", help="Skip publishing")
    args = parser.parse_args()

    # Setup
    print(f"Model: {MODEL}")
    tokenizer = get_tokenizer(MODEL)
    renderer_name = model_info.get_recommended_renderer_name(MODEL)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    print(f"Renderer: {renderer_name}")

    # Prepare training data
    print("Preparing streamed train-only data iterator...")
    train_iter = build_training_iterator(renderer=renderer, max_length=args.max_length)
    train_iter = cycle(train_iter)
    print("  Streaming iterator ready")

    # Create training client
    print(f"Creating LoRA training client (rank={args.rank})...")
    sc = tinker.ServiceClient()
    tc = sc.create_lora_training_client(base_model=MODEL, rank=args.rank)
    print("  Training client ready")

    # Train
    adam_params = types.AdamParams(learning_rate=args.lr, beta1=0.9, beta2=0.95, eps=1e-8)
    print(f"\nTraining for {args.num_steps} steps (batch_size={args.batch_size}, lr={args.lr})...")

    for step in range(args.num_steps):
        # Cycle through data
        batch = [next(train_iter) for _ in range(args.batch_size)]

        fwd_bwd_future = tc.forward_backward(batch, loss_fn="cross_entropy")
        optim_future = tc.optim_step(adam_params)

        fwd_bwd_result = fwd_bwd_future.result()
        optim_future.result()

        # Compute loss
        logprobs = np.concatenate([o["logprobs"].tolist() for o in fwd_bwd_result.loss_fn_outputs])
        weights = np.concatenate([d.loss_fn_inputs["weights"].tolist() for d in batch])
        loss = -np.dot(logprobs, weights) / max(weights.sum(), 1)
        print(f"  Step {step+1}/{args.num_steps} | Loss: {loss:.4f}")

    # Save checkpoint
    print(f"\nSaving checkpoint '{args.checkpoint_name}'...")
    ckpt = tc.save_weights_for_sampler(name=args.checkpoint_name).result()
    checkpoint_path = ckpt.path
    print(f"  Checkpoint saved: {checkpoint_path}")

    # Publish
    if not args.no_publish:
        print("\nPublishing checkpoint...")
        rest_client = sc.create_rest_client()
        rest_client.publish_checkpoint_from_tinker_path(checkpoint_path).result()
        print("  Published successfully!")
    else:
        print("\nSkipping publish (--no_publish).")

    # Save checkpoint info
    info = {
        "checkpoint_path": checkpoint_path,
        "base_model": MODEL,
        "renderer_name": renderer_name,
        "training": {
            "num_steps": args.num_steps,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "lora_rank": args.rank,
            "max_length": args.max_length,
            "datasets": {
                "gsm8k": "openai/gsm8k train",
                "instruction_following": "allenai/tulu-3-sft-mixture train",
                "code": "nvidia/OpenCodeInstruct train",
            },
        },
        "published": not args.no_publish,
    }
    info_path = os.path.join(EVAL_DIR, "checkpoint_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"\nCheckpoint info saved to {info_path}")
    print(f"\nNext: evaluate your checkpoint with")
    print(f"  python -m evaluation.eval_all --checkpoint_path \"{checkpoint_path}\" --base_model {MODEL}")


if __name__ == "__main__":
    main()
