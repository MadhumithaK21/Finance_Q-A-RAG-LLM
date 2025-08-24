
# Script to split a dataset into train and test sets for finetuning
import json, random, os

# Source dataset (JSONL format)
SRC = os.path.join("data", "finetune_dataset_short.jsonl") 
# Output directory for splits
OUT_DIR = os.path.join("data", "ft")
os.makedirs(OUT_DIR, exist_ok=True)

# Load all rows from the source file
with open(SRC, "r", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]

# Shuffle rows for random split (fixed seed for reproducibility)
random.seed(42)
random.shuffle(rows)

# Split: first 10 for test, next 40 for train
test = rows[:10]
train = rows[10:50]  # 40

# Write train split to file
with open(os.path.join(OUT_DIR, "train.jsonl"), "w", encoding="utf-8") as f:
    for r in train:
        f.write(json.dumps(r) + "\n")

# Write test split to file
with open(os.path.join(OUT_DIR, "test.jsonl"), "w", encoding="utf-8") as f:
    for r in test:
        f.write(json.dumps(r) + "\n")

# Print summary of split
print(f"Train: {len(train)} | Test: {len(test)} -> {OUT_DIR}")
