import os, json, time, re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

TEST_FILE = os.path.join("data", "ft", "test.jsonl")
BASE_MODEL = "gpt2"
PEFT_DIR = os.path.join("models", "gpt2-lora-financial")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_test(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def build_prompt(q: str) -> str:
    return f"Q: {q}\nA:"

def extract_amount(txt: str) -> str:
    m = re.search(r"\$\s?-?\d[\d,]*\.?\d*", txt)
    return m.group(0).replace(" ", "") if m else ""

def to_float(val: str):
    if not val:
        return None
    num = re.sub(r"[^0-9.\-]", "", val)  # strip $ and commas
    try:
        return float(num)
    except:
        return None

@torch.no_grad()
def generate_and_score(model, tok, prompt: str, max_new_tokens=12):
    inputs = tok(prompt, return_tensors="pt").to(DEVICE)
    t0 = time.time()
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )
    latency = time.time() - t0
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    gen_text = tok.decode(gen_ids, skip_special_tokens=True)

    # Approximate confidence: average log-prob of generated tokens
    full = out[0]
    logits = model(full.unsqueeze(0)).logits.squeeze(0)
    labels = full.clone()
    labels[:-1] = full[1:]
    labels[-1] = tok.eos_token_id
    start = inputs["input_ids"].shape[1]
    end = full.shape[0]
    if end - start <= 0:
        return gen_text.strip(), float("-inf"), latency
    gen_logits = logits[start-1:end-1]
    gen_labels = labels[start:end]
    log_probs = torch.log_softmax(gen_logits, dim=-1)
    token_lp = log_probs[torch.arange(gen_labels.size(0)), gen_labels]
    avg_lp = token_lp.mean().item() if token_lp.numel() > 0 else float("-inf")
    return gen_text.strip(), avg_lp, latency

def main():
    data = load_test(TEST_FILE)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL).to(DEVICE)
    model = PeftModel.from_pretrained(base, PEFT_DIR).to(DEVICE)
    model.eval()

    exact_correct, tolerant_correct = 0, 0
    lps, lats, details = [], [], []
    for i, row in enumerate(data, 1):
        q = row["instruction"]
        gold = row["output"].strip()
        prompt = build_prompt(q)
        pred_txt, avg_lp, latency = generate_and_score(model, tok, prompt)
        pred = extract_amount(pred_txt)

        # exact match
        if pred == gold:
            exact_correct += 1

        # tolerant numeric match
        gold_val, pred_val = to_float(gold), to_float(pred)
        if gold_val is not None and pred_val is not None:
            if abs(pred_val - gold_val) / max(abs(gold_val), 1e-9) < 0.01:
                tolerant_correct += 1

        lps.append(avg_lp); lats.append(latency)
        details.append({"q": q, "gold": gold, "pred": pred, "lp": avg_lp, "latency": latency})

        print(f"[{i:02d}] gold={gold} pred={pred} lp={avg_lp:.3f} t={latency:.3f}s")

    acc_exact = exact_correct / len(data)
    acc_tol = tolerant_correct / len(data)
    avg_lp = sum(lps)/len(lps) if lps else float("-inf")
    avg_lat = sum(lats)/len(lats) if lats else 0.0

    print("\n=== LORA EVAL RESULTS ===")
    print(f"Exact Accuracy: {acc_exact:.2%}")
    print(f"Tolerant Accuracy (±1%): {acc_tol:.2%}")
    print(f"Avg confidence (avg log-prob): {avg_lp:.3f}")
    print(f"Avg latency (s): {avg_lat:.3f}")

    os.makedirs("runs", exist_ok=True)
    with open("runs/lora_eval.json", "w", encoding="utf-8") as f:
        json.dump({
            "exact_accuracy": acc_exact,
            "tolerant_accuracy": acc_tol,
            "avg_logprob": avg_lp,
            "avg_latency_s": avg_lat,
            "details": details
        }, f, indent=2)

if __name__ == "__main__":
    main()
