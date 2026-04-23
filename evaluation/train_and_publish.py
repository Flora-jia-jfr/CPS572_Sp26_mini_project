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
from datasets import interleave_datasets, load_dataset, IterableDataset

# MODEL = "meta-llama/Llama-3.2-3B"
# MODEL = "meta-llama/Llama-3.2-1B"    # Smaller, faster for development
MODEL = "meta-llama/Llama-3.1-8B"    # Recommended for final submission

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))

def metamath_to_conversation(example):
    return [
        {"role": "user", "content": "Think step by step, then give the final answer.\n\n" + example["query"]},
        {"role": "assistant", "content": example["response"]},
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

# Keywords mapped to IFEval constraint categories.
# We check user messages only — assistant messages don't signal a constraint instruction.
_CONSTRAINT_PHRASES = [
    # Length
    "exactly", "at least", "at most", "no more than", "fewer than", "no fewer than",
    "word count", "number of words", "number of sentences", "number of paragraphs",
    "words long", "sentences long", "paragraphs long",
    "limit your response", "limit the response", "limit your answer",
    "two sentences", "three sentences", "four sentences", "five sentences",
    "two paragraphs", "three paragraphs", "four paragraphs",
    # Format
    "bullet point", "bulleted list", "numbered list", "number each",
    "json format", "in json", "in markdown", "use markdown",
    "in a table", "use a table", "use headers",
    "bold", "italic", "highlight",
    "format your response", "format the response", "format your answer",
    "in the form of", "in the format of",
    "separate each", "separated by",
    # Case
    "all caps", "all capital", "all uppercase", "uppercase only",
    "lowercase only", "lower case", "title case", "capital letters",
    # Keyword inclusion / exclusion
    "must include", "must contain", "include the word", "include the phrase",
    "include the keyword", "use the word", "mention the word",
    "do not use", "don't use", "avoid using", "without using",
    "without the word", "never use",
    "do not include", "don't include", "should not include",
    # Start / end
    "start with", "begin with", "end with", "end your response",
    "start your response", "first word", "last word",
    "ending with", "starting with", "beginning with",
    "postscript", "wrap in", "quotation marks", "in quotes",
    # Response structure
    "your response should", "your answer should", "the response should",
    "response must", "answer must", "your response must",
    "make sure", "ensure that", "ensure your",
    "must be", "should be written", "should be formatted",
    "minimum", "maximum",
    "in a single", "in one paragraph", "in one sentence",
    "step by step", "step-by-step",
    "respond with", "answer in", "write in",
    "using only", "only using",
    "do not repeat", "no repetition",
    "each item", "each point", "each step",
    "provide a list", "include a list",
    "at the end of your", "at the beginning of your",
    # Language
    "in french", "in spanish", "in german", "in japanese",
    "in chinese", "in korean", "in italian", "respond in",
    # Explicit constraint framing
    "following constraints", "following requirements", "following rules",
    "the following format", "constraint:", "requirement:",
]

def has_constraint_instruction(convo: list[dict]) -> bool:
    for msg in convo:
        if msg.get("role") == "user":
            text = msg.get("content", "").lower()
            if any(phrase in text for phrase in _CONSTRAINT_PHRASES):
                return True
    return False


def is_valid_conversation(convo):
    if not convo or len(convo) < 2:
        return False
    has_user = any(m.get("role") == "user" and m.get("content", "").strip() for m in convo)
    has_assistant = any(m.get("role") == "assistant" and m.get("content", "").strip() for m in convo)
    return has_user and has_assistant

SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Follow all instructions precisely, including any formatting, length, "
    "keyword, and style requirements specified by the user."
)

def inject_system_prompt(convo: list[dict]) -> list[dict]:
    """Prepend system prompt if absent; append to existing system message if present."""
    if convo and convo[0].get("role") == "system":
        existing = convo[0]["content"].strip()
        merged = f"{SYSTEM_PROMPT}\n\n{existing}" if existing else SYSTEM_PROMPT
        return [{"role": "system", "content": merged}] + convo[1:]
    return [{"role": "system", "content": SYSTEM_PROMPT}] + convo


METAMATH_FILTERED = os.path.join(EVAL_DIR, "metamath_filtered.jsonl")

def load_metamath_filtered():
    if not os.path.exists(METAMATH_FILTERED):
        raise FileNotFoundError(
            f"Filtered MetaMathQA not found at {METAMATH_FILTERED}. "
            "Run: python evaluation/filter_dataset.py"
        )
    with open(METAMATH_FILTERED) as f:
        for line in f:
            yield json.loads(line)

def build_training_iterator(renderer, max_length):
    metamath = IterableDataset.from_generator(
        lambda: ({"conversation": metamath_to_conversation(ex)} for ex in load_metamath_filtered())
    )

    tulu_full = load_dataset("allenai/tulu-3-sft-mixture", split="train", streaming=True)
    tulu_full = tulu_full.map(lambda ex: {"conversation": tulu_to_conversation(ex)})

    opencode = load_dataset("nvidia/OpenCodeInstruct", split="train", streaming=True)
    opencode = opencode.map(lambda ex: {"conversation": opencode_to_conversation(ex)})

    # Tune3-style mix: full Tulu-3 at 0.45 protects HumanEval; MetaMath for GSM8K.
    mixed = interleave_datasets(
        [metamath, tulu_full, opencode],
        probabilities=[0.35, 0.45, 0.20],
        seed=42,
        stopping_strategy="all_exhausted",
    )

    for ex in mixed:
        convo = ex["conversation"]
        if not is_valid_conversation(convo):
            continue
        try:
            convo = inject_system_prompt(convo)
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
    parser.add_argument("--num_steps", type=int, default=3000, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--rank", type=int, default=64, help="LoRA rank")
    parser.add_argument("--max_length", type=int, default=1536, help="Max token length")
    parser.add_argument("--checkpoint_name", type=str, default="tune6", help="Checkpoint name prefix")
    parser.add_argument("--save_every", type=int, default=500, help="Save an intermediate checkpoint every N steps")
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
    print(f"\nTraining for {args.num_steps} steps (batch_size={args.batch_size}, lr={args.lr}, save_every={args.save_every})...")

    rest_client = sc.create_rest_client() if not args.no_publish else None
    checkpoints = []  # {"step": N, "path": "tinker://..."}
    info_path = os.path.join(EVAL_DIR, "checkpoint_info.json")

    base_info = {
        "base_model": MODEL,
        "renderer_name": renderer_name,
        "training": {
            "num_steps": args.num_steps,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "lora_rank": args.rank,
            "max_length": args.max_length,
            "save_every": args.save_every,
            "datasets": {
                "math": "meta-math/MetaMathQA train, n-gram filtered against GSM8K test (p=0.35, CoT prefix)",
                "instruction_following": "allenai/tulu-3-sft-mixture train (p=0.45, unfiltered)",
                "code": "nvidia/OpenCodeInstruct train (p=0.20)",
            },
        },
        "published": not args.no_publish,
    }

    def save_and_publish(name, step):
        print(f"\n  Saving checkpoint '{name}' at step {step}...")
        ckpt = tc.save_weights_for_sampler(name=name).result()
        path = ckpt.path
        print(f"    Saved: {path}")
        if rest_client:
            rest_client.publish_checkpoint_from_tinker_path(path).result()
            print(f"    Published.")
        checkpoints.append({"step": step, "name": name, "path": path})
        # Write immediately so paths are available mid-training
        info = {**base_info, "checkpoint_path": path, "intermediate_checkpoints": checkpoints}
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)
        print(f"    checkpoint_info.json updated.")
        return path

    for step in range(args.num_steps):
        batch = [next(train_iter) for _ in range(args.batch_size)]

        fwd_bwd_future = tc.forward_backward(batch, loss_fn="cross_entropy")
        optim_future = tc.optim_step(adam_params)

        fwd_bwd_result = fwd_bwd_future.result()
        optim_future.result()

        logprobs = np.concatenate([o["logprobs"].tolist() for o in fwd_bwd_result.loss_fn_outputs])
        weights = np.concatenate([d.loss_fn_inputs["weights"].tolist() for d in batch])
        loss = -np.dot(logprobs, weights) / max(weights.sum(), 1)
        print(f"  Step {step+1}/{args.num_steps} | Loss: {loss:.4f}")

        # Save intermediate checkpoint (skip final step — saved separately below)
        if (step + 1) % args.save_every == 0 and (step + 1) < args.num_steps:
            save_and_publish(f"{args.checkpoint_name}_step{step+1}", step + 1)

    # Save final checkpoint
    final_path = save_and_publish(args.checkpoint_name, args.num_steps)

    print(f"\nCheckpoint info saved to {info_path}")
    print(f"\nEvaluate each checkpoint to find the best one:")
    for ckpt in checkpoints:
        print(f"  step {ckpt['step']:>5}: python evaluation/eval_all.py --checkpoint_path \"{ckpt['path']}\" --base_model {MODEL}")


if __name__ == "__main__":
    main()
