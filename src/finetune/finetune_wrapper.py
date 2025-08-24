
# Regular expressions and PyTorch
import re
import torch
# HuggingFace Transformers and PEFT for LoRA
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Directory containing the LoRA adapter
MODEL_DIR = "models/gpt2-lora-financial"
# Name of the base model
BASE_MODEL = "gpt2"
# Use GPU if available, else CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load the fine-tuned GPT-2 model with LoRA adapters and tokenizer
def load_finetuned_model():
    """Load fine-tuned GPT-2 model with LoRA adapters and tokenizer."""
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    model = PeftModel.from_pretrained(base_model, MODEL_DIR)
    model.eval()
    model.to(DEVICE)
    return model, tok

# Guardrail to filter out unsafe or irrelevant input questions
def apply_input_guardrail(question: str) -> bool:
    harmful_keywords = ["kill", "bomb", "hate", "sex", "violence"]
    if not question.strip():
        return False
    if any(word in question.lower() for word in harmful_keywords):
        return False
    return True

# Guardrail to ensure output contains a financial figure
def apply_output_guardrail(answer: str) -> str:
    if not re.search(r"[\d$]", answer):
        return "I could not find a valid financial figure for your question."
    return answer

# Generate an answer using the fine-tuned model, applying input/output guardrails
@torch.no_grad()
def get_finetuned_answer(model, tok, question: str) -> str:
    """Run inference on fine-tuned GPT-2 with LoRA adapters and guardrails."""
    # Check input guardrail
    if not apply_input_guardrail(question):
        return "Input rejected by guardrail (irrelevant or unsafe)."
    # Build prompt and tokenize
    prompt = f"Q: {question}\nA:"
    inputs = tok(prompt, return_tensors="pt").to(DEVICE)
    # Generate model output
    out = model.generate(
        **inputs,
        max_new_tokens=16,
        do_sample=False,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )
    # Decode generated answer (excluding prompt)
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    raw_answer = tok.decode(gen_ids, skip_special_tokens=True).strip()
    # Apply output guardrail
    return apply_output_guardrail(raw_answer)
