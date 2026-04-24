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
import random
import time
import math

import numpy as np
import tinker
from tinker import types
from tinker import Datum
from tinker_cookbook import model_info, renderers
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer

from torch.utils.data import Dataset
from datasets import load_dataset
from itertools import accumulate

# MODEL = "meta-llama/Llama-3.2-3B"
# MODEL = "meta-llama/Llama-3.2-1B"    # Smaller, faster for development
MODEL = "meta-llama/Llama-3.1-8B"    # Recommended for final submission

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))

# TODO: TOY DATA, replace with your own training data
DEMO_CONVERSATIONS = [
    [
        {"role": "user", "content": "What is 15 + 27?"},
        {"role": "assistant", "content": "15 + 27 = 42"},
    ],
    [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
    ],
    [
        {"role": "user", "content": "Write a Python function that returns the sum of two numbers."},
        {"role": "assistant", "content": "def add(a, b):\n    return a + b"},
    ],
    [
        {"role": "user", "content": "What is 8 * 7?"},
        {"role": "assistant", "content": "8 * 7 = 56"},
    ],
    [
        {"role": "user", "content": "Translate 'hello' to Spanish."},
        {"role": "assistant", "content": "Hola"},
    ],
    [
        {"role": "user", "content": "What is the square root of 144?"},
        {"role": "assistant", "content": "The square root of 144 is 12."},
    ],
    [
        {"role": "user", "content": "Write a Python function to check if a number is even."},
        {"role": "assistant", "content": "def is_even(n):\n    return n % 2 == 0"},
    ],
    [
        {"role": "user", "content": "List the first 5 prime numbers."},
        {"role": "assistant", "content": "The first 5 prime numbers are: 2, 3, 5, 7, 11."},
    ],
]

### DATASETS

# Sampler randomly picks a datum object with replacement from a distribution.
class Sampler(Dataset):
    def __init__(
        self,
        datasets: dict,
        weights: dict,
        total_size: int,
        seed: int = 42,
    ):
        """
        datasets: dict of {name: list_of_examples}
        weights: dict of {name: float} (should sum to 1.0)
        total_size: virtual length of dataset
        """
        assert set(datasets.keys()) == set(weights.keys())

        self.datasets = datasets
        self.names = list(datasets.keys())
        self.weights = [weights[name] for name in self.names]
        self.total_size = total_size

        self.rng = random.Random(seed)

        # Precompute cumulative distribution for fast sampling
        total = sum(self.weights)
        probs = [w / total for w in self.weights]

        self.cdf = []
        cumsum = 0.0
        for p in probs:
            cumsum += p
            self.cdf.append(cumsum)

    def __len__(self):
        return self.total_size

    def _sample_dataset(self):
        r = self.rng.random()
        for name, threshold in zip(self.names, self.cdf):
            if r <= threshold:
                return name
        return self.names[-1]

    def sample(self):
        # pick dataset
        name = self._sample_dataset()
        dataset = self.datasets[name]
        example = dataset[self.rng.randint(0, len(dataset) - 1)]

        return example

    def tolist(self):
        data = [self.getitem(0) for i in range(self.total_size)]
        return data

def format_tutu(example):
    instruction = example.get("messages")[0]
    response = example.get("messages")[1]
    return [
            instruction, response
        ]

def format_gsm8k(example):
    return [
            {
                "role": "user",
                "content": example['question']
            },
            {
                "role": "assistant",
                "content": example["answer"]
            }
        ]

def format_magicoder(example): #sandra added
    return [
            {
                "role": "user",
                "content": example['problem']
            },
            {
                "role": "assistant",
                "content": example["solution"]
            }
        ]

def format_mbpp(example):
    return [
            {
                "role": "user",
                "content": example['text']
            },
            {
                "role": "assistant",
                "content": example['code']
            }
        ]

def format_opencode(example):
    return [
            {
                "role": "user",
                "content": example['input']
            },
            {
                "role": "assistant",
                "content": example['output']
            }
        ]


def main():
    parser = argparse.ArgumentParser(description="Train, save, and publish a checkpoint")
    parser.add_argument("--num_steps", type=int, default=600, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--rank", type=int, default=64, help="LoRA rank")
    parser.add_argument("--checkpoint_name", type=str, default="demo", help="Checkpoint name")
    parser.add_argument("--no_publish", action="store_true", help="Skip publishing")
    parser.add_argument("--val_every", type=int, default=10, help="Validate every N steps")
    parser.add_argument("--val_batch_size", type=int, default=64, help="Validation batch size")
    parser.add_argument("--early_stopping", type=bool, default=True, help="Early Stopping")
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience (number of validations to wait for improvement)")
    args = parser.parse_args()

    # Setup
    print(f"Model: {MODEL}")
    tokenizer = get_tokenizer(MODEL)
    renderer_name = model_info.get_recommended_renderer_name(MODEL)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    print(f"Renderer: {renderer_name}")

    # Prepare training data
    print("Preparing training data...")

    def split_dataset(all_data, verbose = True):
        # 75/25 train/validation split
        indices = random.sample(range(len(all_data)), len(all_data))
        train_size = int(0.75 * len(all_data))
        train_data = [all_data[i] for i in indices[:train_size]]
        val_data = [all_data[i] for i in indices[train_size:]]
        if verbose:
            print(f"  Total Data: {len(all_data)} | Train: {len(train_data)} | Val: {len(val_data)}")
        return train_data, val_data
    
    def to_datum(dataset):
        all_data = []
        for i in range(len(dataset)):
            convo = dataset[i]
            datum = conversation_to_datum(
                convo, renderer, max_length=512, train_on_what=renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES
            )
            all_data.append(datum)
        return all_data

    def split_by_difficulty(dataset, name):
        if name == "tutu":
            # tutu ifeval dataset
            easy, medium, hard = [],[],[]
            for data in dataset:
                difficulty = len(data.get("constraints") or [])
                if difficulty < 2:
                    easy.append(format_tutu(data))
                elif difficulty == 2:
                    medium.append(format_tutu(data))
                else:
                    hard.append(format_tutu(data))
            # print(f"{len(easy)}, {len(medium), {len(hard)}}")
            # print(easy[1])
            # print(medium[1])
            # print(hard[1])
            easy, medium, hard = to_datum(easy), to_datum(medium), to_datum(hard)
            return easy, medium, hard
        raise NotImplementedError()

    def process_dataset(name, split = False):
        if name == "tutu":
            ds = load_dataset("allenai/tulu-3-sft-personas-instruction-following", split = "train", streaming=True)
            dataset_tutu = list(ds.take(5000))
            if split:
                easy, medium, hard = split_by_difficulty(dataset_tutu, "tutu")
                return {"easy": easy, "medium": medium, "hard": hard}
            dataset_tutu = to_datum([format_tutu(s) for s in dataset_tutu])
            return dataset_tutu
        if name == "tutu-gsm":
            ds = load_dataset("allenai/tulu-3-sft-personas-math-grade-filtered", split = "train", streaming=True)
            dataset_tutu = list(ds.take(2000))
            dataset_tutu = to_datum([format_tutu(s) for s in dataset_tutu])
            return dataset_tutu
        if name == "gsm8k":
            dataset_gsm = load_dataset("gsm8k", "main", split = "train")
            # print([format_gsm8k(s) for s in dataset_gsm][:10])
            dataset_gsm = to_datum([format_gsm8k(s) for s in dataset_gsm])
            return dataset_gsm
        if name == "mbpp":
            dataset_mbpp = load_dataset("mbpp", split = "train")
            # print([format_mbpp(s) for s in dataset_mbpp][:10])
            dataset_mbpp = to_datum([format_mbpp(s) for s in dataset_mbpp])
            return dataset_mbpp
        if name == "opencode":
            ds = load_dataset("nvidia/OpenCodeInstruct", split = "train", streaming = True)
            dataset = list(ds.take(3000))
            dataset = to_datum([format_opencode(s) for s in dataset if float(s["average_test_score"]) > 0.8])
            return dataset
        if name == "metamath":
            ds = load_dataset("meta-math/MetaMathQA", split="train", streaming=True)
            dataset = list(ds.take(20000))
            dataset = to_datum([
                [{"role": "user", "content": x["query"]},
                 {"role": "assistant", "content": x["response"]}]
                for x in dataset
            ])
            return dataset
        if name == "magicoder":
            ds = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train", streaming=True)
            dataset = list(ds.take(3000))
            dataset = to_datum([format_magicoder(s) for s in dataset])
            return dataset

        raise ValueError("Invalid dataset name.")
    dataset_gsm = process_dataset("gsm8k")
    dataset_coding = process_dataset("mbpp")

    dataset_tutu_split = process_dataset("tutu", split = True)
    dataset_gsm_additional = process_dataset("tutu-gsm")
    dataset_metamath = process_dataset("metamath")

    gsm_combined = dataset_gsm + dataset_gsm_additional + dataset_metamath

    def sample_from_dataset(dataset, value):
        passes = value // len(dataset)
        return dataset * passes + random.sample(dataset, value - passes * len(dataset))

    # TODO: tune these
    # START OF WEIGHTS
    total_samples = 5000
    num_stages = 3
    
    #data_weights = [0.15, 0.40, 0.45] # newest
    #data_weights = [0.25, 0.55, 0.2] # newer -- got best results w/ this one!!!!!
    data_weights = [0.20, 0.45, 0.35] # more stage 2 for code
    #data_weights = [0.3, 0.5, 0.2] # proportion of training samples allotted to each learning stage

    # proportion of tasks in each stage

    # stage_weights = [ # newest -- untested
    #     # Stage 0: lighter warmup
    #     {"if-eval easy": 0.30, "if-eval medium": 0.10, "if-eval hard": 0.00, "gsm8k": 0.45, "humaneval": 0.15},
    #     # Stage 1: boost GSM8K, grow humaneval earlier
    #     {"if-eval easy": 0.00, "if-eval medium": 0.10, "if-eval hard": 0.05, "gsm8k": 0.60, "humaneval": 0.25},
    #     # Stage 2: heavy code push, hard IF, keep some math for retention
    #     {"if-eval easy": 0.00, "if-eval medium": 0.00, "if-eval hard": 0.25, "gsm8k": 0.15, "humaneval": 0.60},
    # ]
    stage_weights = [ #newer -- got best results w this version!!!!
        # Stage 0: similar warmup
        {"if-eval easy": 0.35, "if-eval medium": 0.10, "if-eval hard": 0.00, "gsm8k": 0.40, "humaneval": 0.15},
        # Stage 1: GSM8K dominant but grow code earlier
        {"if-eval easy": 0.00, "if-eval medium": 0.10, "if-eval hard": 0.10, "gsm8k": 0.55, "humaneval": 0.25},
        # Stage 2: finish with code + hard IF, minimal math to avoid forgetting
        {"if-eval easy": 0.00, "if-eval medium": 0.00, "if-eval hard": 0.30, "gsm8k": 0.20, "humaneval": 0.50},
    ]

    # stage_weights = [

    #     # stage 0
    #     {"if-eval easy": 0.4,
    #     "if-eval medium": 0.1,
    #     "if-eval hard": 0,
    #     "gsm8k": 0.3,
    #     "humaneval": 0.2},

    #     # stage 1
    #     {"if-eval easy": 0,
    #     "if-eval medium": 0.1,
    #     "if-eval hard": 0.2,
    #     "gsm8k": 0.5,
    #     "humaneval": 0.2},

    #     # stage 2
    #     {"if-eval easy": 0,
    #     "if-eval medium": 0,
    #     "if-eval hard": 0.3,
    #     "gsm8k": 0.3,
    #     "humaneval": 0.4}

    # ]

    # END OF WEIGHTS

    stage_samples = [total_samples * dw for dw in data_weights]
    train_sets = []
    val_sets = []

    for s in range(num_stages):
        stage_dataset = (sample_from_dataset(dataset_tutu_split["easy"], int(stage_samples[s] * stage_weights[s]["if-eval easy"])) 
            + sample_from_dataset(dataset_tutu_split["medium"], int(stage_samples[s] * stage_weights[s]["if-eval medium"])) 
            + sample_from_dataset(dataset_tutu_split["hard"], int(stage_samples[s] * stage_weights[s]["if-eval hard"])) 
            + sample_from_dataset(gsm_combined, int(stage_samples[s] * stage_weights[s]["gsm8k"])) 
            + sample_from_dataset(dataset_coding, int(stage_samples[s] * stage_weights[s]["humaneval"])))
        random.shuffle(stage_dataset)
        train_stage, val_stage = split_dataset(stage_dataset)
        train_sets.append(train_stage)
        val_sets.append(val_stage)


    # all_data = sample_from_dataset(dataset_tutu, 600) + sample_from_dataset(dataset_gsm, 800) + sample_from_dataset(dataset_coding, 600) 
    # random.shuffle(all_data)

    # train_data, val_data = split_dataset(all_data)
    # val_batches = [val_data[i : i + args.val_batch_size] for i in range(0, len(val_data), args.val_batch_size)]

    val_batches_bystage = [[val_data[i : i + args.val_batch_size] for i in range(0, len(val_data), args.val_batch_size)] 
                    for val_data in val_sets]

    # Create training client
    print(f"Creating LoRA training client (rank={args.rank})...")
    sc = tinker.ServiceClient()
    tc = sc.create_lora_training_client(base_model=MODEL, rank=args.rank)
    print("  Training client ready")

    # Train
    adam_params = types.AdamParams(learning_rate=args.lr, beta1=0.9, beta2=0.95, eps=1e-8)
    print(f"\nTraining for {args.num_steps} steps (batch_size={args.batch_size}, lr={args.lr})...")
    
    #Early Stopping Variables
    best_val_loss = float("inf")
    patience_counter = 0

    max_attained_stage = -1
    train_data = None
    val_batches = None
    cum_weights = [0]+list(accumulate(data_weights))
    step = -1
    stage = -1

    train_losses = []
    val_losses = []

    while step < args.num_steps:
        step += 1
        # Cycle through training data
        # start = (step * args.batch_size) % len(train_data)
        # batch = [train_data[i % len(train_data)] for i in range(start, start + args.batch_size)]
        # batch = [train_data[random.randint(0, len(train_data)-1)] for _ in range(batch_size)]
        
        ratio = (step+1)/args.num_steps
        if ratio > cum_weights[stage + 1]:
            stage += 1
            if stage >= num_stages:
                break
            print(f"Starting training curriculum: {stage}")
            train_data = train_sets[stage]
            val_batches = val_batches_bystage[stage]
            val_data = val_sets[stage]

        batch = random.sample(train_data, args.batch_size) 
        
        fwd_bwd_future = tc.forward_backward(batch, loss_fn="cross_entropy")
        optim_future = tc.optim_step(adam_params)

        fwd_bwd_result = fwd_bwd_future.result()
        optim_future.result()
        
        train_total = 0.0
        train_count = 0.0
        for logprob_output, datum in zip(fwd_bwd_result.loss_fn_outputs, batch):
            weights = np.asarray(datum.loss_fn_inputs["weights"].tolist())
            logprobs = np.asarray(logprob_output["logprobs"].tolist())
            train_total += float(np.dot(logprobs, weights))
            train_count += float(weights.sum())
        train_loss = -train_total / max(train_count, 1.0)
        
        train_losses.append({"step": step+1, "stage": stage, "train_loss": train_loss})

        
        print(f"Curriculum {stage} | Step {step+1}/{args.num_steps} | Train Loss: {train_loss:.4f}")
        
        # Validation
        should_validate = (((step + 1) % args.val_every == 0) or (step == args.num_steps - 1)) and len(val_batches) > 0
        if should_validate:
            ###
            val_block_start = time.perf_counter()
            print(f"  Step {step+1}/{args.num_steps} | validation block start @ {time.strftime('%H:%M:%S')}")
            ###
            val_total = 0.0
            val_count = 0.0
            for val_batch in val_batches:
                val_result = tc.forward(val_batch, loss_fn="cross_entropy").result()

                batch_logprobs = np.concatenate([o["logprobs"].tolist() for o in val_result.loss_fn_outputs])
                batch_weights = np.concatenate([d.loss_fn_inputs["weights"].tolist() for d in val_batch])
                val_total += float(np.dot(batch_logprobs, batch_weights))
                val_count += float(batch_weights.sum())

            val_loss = -val_total / max(val_count, 1.0)
            val_losses.append({"step": step+1, "stage": stage, "val_loss": val_loss})

            ###
            val_block_elapsed = time.perf_counter() - val_block_start
            ###
        
            print(
                f"  Step {step+1}/{args.num_steps} | Val Loss: {val_loss:.4f}"
                f" | Val Time: {val_block_elapsed:.2f}s | Val Examples: {len(val_data)}"
            )

            # Early Stopping (optional)
            if args.early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    print(f"  No improvement in val_loss. Patience counter: {patience_counter}/{args.patience}")

                if patience_counter >= args.patience:
                    print(f"  Early stopping at step {step+1} (val_loss={val_loss:.4f} did not improve over best_loss={best_val_loss:.4f})")
                    print(f"  Ending curriculum {stage}")
                    if stage == num_stages - 1:
                        break
                    else:
                        stage += 1
                        print(f"Starting training curriculum: {stage}")
                        train_data = train_sets[stage]
                        val_batches = val_batches_bystage[stage]
                        val_data = val_sets[stage]
                        patience_counter = 0
                        best_val_loss = float("inf")
            
        ###
        else: 
            val_loss = None

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
        },
        "published": not args.no_publish,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
    info_path = os.path.join(EVAL_DIR, "checkpoint_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"\nCheckpoint info saved to {info_path}")
    print(f"\nNext: evaluate your checkpoint with")
    print(f"  python -m evaluation.eval_all --checkpoint_path \"{checkpoint_path}\" --base_model {MODEL}")


if __name__ == "__main__":
    main()

