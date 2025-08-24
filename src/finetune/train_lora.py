# src/finetune/train_lora.py
import os
import json
from datasets import load_dataset
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datetime import datetime

# -------- CONFIG --------
BASE_MODEL = "gpt2"                      # small base model
TRAIN_FILE = os.path.join("data", "ft", "train.jsonl")
OUTPUT_DIR = os.path.join("models", "gpt2-lora-financial")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# LoRA hyperparameters (log these)
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ["c_attn"]  # GPT-2 style attention module name

# Training hyperparams (log these)
LEARNING_RATE = 5e-4
BATCH_SIZE = 8
EPOCHS = 5
MAX_LENGTH = 256
GRAD_ACCUM = 1

# ------------------------

def format_example(example):
    q = example["instruction"].strip()
    a = example["output"].strip()
    text = f"Q: {q}\nA: {a}"
    return {"text": text}

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Load dataset (jsonl each row has instruction, output)
    ds = load_dataset("json", data_files=TRAIN_FILE)["train"]
    ds = ds.map(format_example, remove_columns=ds.column_names)

    # Tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=MAX_LENGTH)

    ds_tok = ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    model.to(DEVICE)

    # Prepare model for k-bit training if using 8-bit (optional)
    # model = prepare_model_for_kbit_training(model)  # only if using bitsandbytes 8-bit

    # Configure LoRA
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    # Data collator for causal LM
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # Training args
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none",
        remove_unused_columns=False,
        push_to_hub=False,
    )

    # Save hyperparams for reproducibility
    with open(os.path.join(OUTPUT_DIR, "hyperparams.json"), "w") as f:
        json.dump({
            "base_model": BASE_MODEL,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "target_modules": TARGET_MODULES,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "max_length": MAX_LENGTH,
            "device": DEVICE,
        }, f, indent=2)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_tok,
        data_collator=data_collator,
    )

    trainer.train()
    # Save the full peft adapter
    model.save_pretrained(OUTPUT_DIR)
    print("Saved LoRA/PEFT weights to", OUTPUT_DIR)

if __name__ == "__main__":
    main()
