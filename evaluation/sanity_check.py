# Quick format check - run this standalone
from datasets import load_dataset
ds = load_dataset("gsm8k", "main", split="train")
print(ds[0]["answer"])  # Should end with "#### <number>"

# Then check what your model actually outputs on a GSM8K prompt
# and whether the eval script's regex matches it