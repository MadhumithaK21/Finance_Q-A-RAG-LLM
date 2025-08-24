import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
 
MODEL_DIR = "models/gpt2-lora-financial"
BASE_MODEL = "gpt2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
 
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
 
def apply_input_guardrail(question: str) -> bool:
    harmful_keywords = ["kill", "bomb", "hate", "sex", "violence"]
    if not question.strip():
        return False
    if any(word in question.lower() for word in harmful_keywords):
        return False
    return True
 
def apply_output_guardrail(answer: str) -> str:
    if not re.search(r"[\d$]", answer):
        return "I could not find a valid financial figure for your question."
    return answer
 
@torch.no_grad()
def get_finetuned_answer(model, tok, question: str) -> str:
    """Run inference on fine-tuned GPT-2 with LoRA adapters and guardrails."""
    if not apply_input_guardrail(question):
        return "Input rejected by guardrail (irrelevant or unsafe)."
    prompt = f"Q: {question}\nA:"
    inputs = tok(prompt, return_tensors="pt").to(DEVICE)
    out = model.generate(
        **inputs,
        max_new_tokens=16,
        do_sample=False,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    raw_answer = tok.decode(gen_ids, skip_special_tokens=True).strip()
    return apply_output_guardrail(raw_answer)
