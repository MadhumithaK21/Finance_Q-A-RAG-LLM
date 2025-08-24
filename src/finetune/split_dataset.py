import json, random, os

SRC = os.path.join("data", "finetune_dataset_short.jsonl") 
OUT_DIR = os.path.join("data", "ft")
os.makedirs(OUT_DIR, exist_ok=True)

with open(SRC, "r", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]

random.seed(42)
random.shuffle(rows)

test = rows[:10]
train = rows[10:50]  # 40

with open(os.path.join(OUT_DIR, "train.jsonl"), "w", encoding="utf-8") as f:
    for r in train:
        f.write(json.dumps(r) + "\n")

with open(os.path.join(OUT_DIR, "test.jsonl"), "w", encoding="utf-8") as f:
    for r in test:
        f.write(json.dumps(r) + "\n")

print(f"Train: {len(train)} | Test: {len(test)} -> {OUT_DIR}")
