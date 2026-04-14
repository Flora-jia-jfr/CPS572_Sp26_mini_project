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

import numpy as np
import tinker
from tinker import types
from tinker import Datum
from tinker_cookbook import model_info, renderers
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer

from torch.utils.data import Dataset
from datasets import load_dataset

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
    parser.add_argument("--num_steps", type=int, default=10, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--checkpoint_name", type=str, default="demo", help="Checkpoint name")
    parser.add_argument("--no_publish", action="store_true", help="Skip publishing")
    parser.add_argument("--val_every", type=int, default=5, help="Validate every N steps")
    parser.add_argument("--val_batch_size", type=int, default=64, help="Validation batch size")
    parser.add_argument("--early_stopping", type=bool, default=True, help="Early Stopping")
    parser.add_argument("--patience", type=int, default=2, help="Early stopping patience (number of validations to wait for improvement)")
    args = parser.parse_args()

    # Setup
    print(f"Model: {MODEL}")
    tokenizer = get_tokenizer(MODEL)
    renderer_name = model_info.get_recommended_renderer_name(MODEL)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    print(f"Renderer: {renderer_name}")

    # Prepare training data
    print("Preparing training data...")

    def split_dataset(all_data):
        # 75/25 train/validation split
        indices = random.sample(range(len(all_data)), len(all_data))
        train_size = int(0.75 * len(all_data))
        train_data = [all_data[i] for i in indices[:train_size]]
        val_data = [all_data[i] for i in indices[train_size:]]
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

    def process_dataset(name):
        if name == "tutu":
            ds = load_dataset("allenai/tulu-3-sft-mixture", split = "train", streaming=True)
            dataset_tutu = list(ds.take(1000))
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
            dataset = list(ds.take(1000))
            dataset = to_datum([format_opencode(s) for s in dataset])
            return dataset

        raise ValueError("Invalid dataset name.")
    dataset_tutu = process_dataset("tutu") #tutu for ifEval
    dataset_gsm = process_dataset("gsm8k") #gsm8k
    # dataset_coding_train, dataset_coding_val = process_dataset("mbpp") #mbpp dataset for HumanEval
    dataset_coding = process_dataset("opencode") #opencode dataset for HumanEval

    # train_data = MixedDataset(
    #     datasets = {
    #         "tutu": dataset_tutu_train,
    #         "gsm8k": dataset_gsm_train,
    #         "mbpp": dataset_coding_train
    #     },
    #     weights = {
    #         "tutu": 0.3,
    #         "gsm8k": 0.4,
    #         "mbpp": 0.3
    #     },
    #     total_size = 1500
    # )

    # val_data = MixedDataset(
    #     datasets = {
    #         "tutu": dataset_tutu_val,
    #         "gsm8k": dataset_gsm_val,
    #         "mbpp": dataset_coding_val
    #     },
    #     weights = {
    #         "tutu": 0.3,
    #         "gsm8k": 0.4,
    #         "mbpp": 0.3
    #     },
    #     total_size = 500
    # )

    def sample_from_dataset(dataset, value):
        passes = value // len(dataset)
        return dataset * passes + random.sample(dataset, value - passes * len(dataset))

    all_data = sample_from_dataset(dataset_tutu, 600) + sample_from_dataset(dataset_gsm, 800) + sample_from_dataset(dataset_coding, 600) 
    random.shuffle(all_data)
    # all_data = []
    # for i in range(len(DEMO_CONVERSATIONS)):
    #     convo = DEMO_CONVERSATIONS[i]
    #     datum = conversation_to_datum(
    #         convo, renderer, max_length=512, train_on_what=renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES
    #     )
    #     all_data.append(datum)

    train_data, val_data = split_dataset(all_data)
    val_batches = [val_data[i : i + args.val_batch_size] for i in range(0, len(val_data), args.val_batch_size)]

    # Create training client
    print(f"Creating LoRA training client (rank={args.rank})...")
    sc = tinker.ServiceClient()
    tc = sc.create_lora_training_client(base_model=MODEL, rank=args.rank)
    print("  Training client ready")

    # Train
    adam_params = types.AdamParams(learning_rate=args.lr, beta1=0.9, beta2=0.95, eps=1e-8)
    print(f"\nTraining for {args.num_steps} steps (batch_size={args.batch_size}, lr={args.lr})...")
    
    #Early Stopping Variables
    prev_val_loss = float("inf")
    patience_counter = 0    

    for step in range(args.num_steps):
        # Cycle through training data
        # start = (step * args.batch_size) % len(train_data)
        # batch = [train_data[i % len(train_data)] for i in range(start, start + args.batch_size)]
        # batch = [train_data[random.randint(0, len(train_data)-1)] for _ in range(batch_size)]
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
        
        print(f" Step {step+1}/{args.num_steps} | Train Loss: {train_loss:.4f}")
        
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
            ###
            val_block_elapsed = time.perf_counter() - val_block_start
            ###
        
            print(
                f"  Step {step+1}/{args.num_steps} | Val Loss: {val_loss:.4f}"
                f" | Val Time: {val_block_elapsed:.2f}s | Val Examples: {len(val_data)}"
            )

            # Early Stopping (optional)
            if args.early_stopping:
                if val_loss <= prev_val_loss:
                    prev_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    prev_val_loss = val_loss  # Update previous loss even if no improvement, to make sure high val wasn't a fluke
                    print(f"  No improvement in val_loss. Patience counter: {patience_counter}/{args.patience}")
                
                if patience_counter >= args.patience:
                    print(f"  Early stopping at step {step+1} (val_loss={val_loss:.4f} did not improve over best_loss={prev_val_loss:.4f})")
                    break
            
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
    }
    info_path = os.path.join(EVAL_DIR, "checkpoint_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"\nCheckpoint info saved to {info_path}")
    print(f"\nNext: evaluate your checkpoint with")
    print(f"  python -m evaluation.eval_all --checkpoint_path \"{checkpoint_path}\" --base_model {MODEL}")


if __name__ == "__main__":
    main()

