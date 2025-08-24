import os, json
from datasets import load_dataset
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from transformers.adapters import AdapterConfig

TRAIN_PATH = os.path.join("data", "ft", "train.jsonl")
MODEL_NAME = "gpt2"
OUTPUT_DIR = "models/gpt2-financial-adapter"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def format_example(example):
    # Simple instruction-following format
    q = example["instruction"].strip()
    a = example["output"].strip()
    # Keep answers short, we train to emit just the number after "A:"
    text = f"Q: {q}\nA: {a}"
    return {"text": text}

def main():
    ds = load_dataset("json", data_files={"train": TRAIN_PATH})
    ds = ds.map(format_example, remove_columns=ds["train"].column_names)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def tokenize(batch):
        return tok(batch["text"], truncation=True, max_length=256)

    ds_tok = ds.map(tokenize, batched=True, remove_columns=["text"])

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.to(DEVICE)

    # ----- Adapter config (Pfeiffer) -----
    config = AdapterConfig.load("pfeiffer", reduction_factor=16, non_linearity="relu")
    model.add_adapter("fin_ft", config=config)
    model.train_adapter("fin_ft")

    # ----- Training args (log hyperparams here) -----
    lr = 5e-4
    bsz = 8
    epochs = 5

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=bsz,
        gradient_accumulation_steps=1,
        learning_rate=lr,
        num_train_epochs=epochs,
        logging_steps=20,
        save_steps=200,
        save_total_limit=2,
        bf16=False,
        fp16=False,
        report_to="none",
        remove_unused_columns=False,
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_tok["train"],
        data_collator=collator,
    )

    # ----- Log setup -----
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "hyperparams.json"), "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "adapter": "pfeiffer",
            "reduction_factor": 16,
            "learning_rate": lr,
            "batch_size": bsz,
            "epochs": epochs,
            "device": DEVICE,
        }, f, indent=2)

    trainer.train()

    # Save adapter only (small)
    model.save_adapter(OUTPUT_DIR, "fin_ft")
    print(f"Saved adapter to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
