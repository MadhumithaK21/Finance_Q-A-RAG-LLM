
# Script to train a GPT-2 model with adapters (Pfeiffer) for financial Q&A
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


# Path to training data
TRAIN_PATH = os.path.join("data", "ft", "train.jsonl")
# Name of the base model
MODEL_NAME = "gpt2"
# Output directory for the adapter
OUTPUT_DIR = "models/gpt2-financial-adapter"

# Use GPU if available, else CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Format each example as a prompt/response pair for instruction tuning
def format_example(example):
    # Simple instruction-following format
    q = example["instruction"].strip()
    a = example["output"].strip()
    # Keep answers short, we train to emit just the number after "A:"
    text = f"Q: {q}\nA: {a}"
    return {"text": text}


# Main training routine
def main():
    # Load dataset from JSONL file
    ds = load_dataset("json", data_files={"train": TRAIN_PATH})
    # Format each example for instruction tuning
    ds = ds.map(format_example, remove_columns=ds["train"].column_names)

    # Load tokenizer and set pad token if needed
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Tokenize the dataset
    def tokenize(batch):
        return tok(batch["text"], truncation=True, max_length=256)

    ds_tok = ds.map(tokenize, batched=True, remove_columns=["text"])

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.to(DEVICE)

    # ----- Adapter config (Pfeiffer) -----
    # Add and train a Pfeiffer adapter for financial finetuning
    config = AdapterConfig.load("pfeiffer", reduction_factor=16, non_linearity="relu")
    model.add_adapter("fin_ft", config=config)
    model.train_adapter("fin_ft")

    # ----- Training args (log hyperparams here) -----
    # Set training hyperparameters
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

    # Data collator for language modeling
    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    # Trainer setup
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_tok["train"],
        data_collator=collator,
    )

    # ----- Log setup -----
    # Save hyperparameters for reproducibility
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

    # Start training
    trainer.train()

    # Save adapter only (small)
    model.save_adapter(OUTPUT_DIR, "fin_ft")
    print(f"Saved adapter to {OUTPUT_DIR}")

# Run training if script is executed directly
if __name__ == "__main__":
    main()
