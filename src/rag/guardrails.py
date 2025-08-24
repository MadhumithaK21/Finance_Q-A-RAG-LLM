import re
from typing import List, Tuple, Dict

# ---- Input guardrail ----

FINANCIAL_KEYWORDS = {
    "statement","balance","assets","liabilities","equity","revenue","income",
    "cash","expenses","benefits","securities","investments","audit","appendix",
    "financial","notes","footnotes","net","profit","loss","comprehensive","operations",
    "disclosures","accrual","payable","receivable","dividend","interest","depreciation",
    "amortization","tax","fund","plan","participant","pension","contribution"
}

HARMFUL_BLOCKLIST = [
    r"\bhow\s+to\s+hack\b",
    r"\bmalware|ransomware|keylogger|ddos\b",
    r"\bmake\s+bomb|explosive\b",
    r"\bsuicide|self[-\s]?harm|kill\s+myself\b",
    r"\bcredit\s*card\s*number\b",
    r"\bpasswords?\b",
    r"\bssn\b|\bsocial\s+security\s+number\b"
]

PROFANITY = [
    r"\b(fuck|shit|bitch|bastard|asshole)\b"
]


def input_guardrail_check(query: str) -> Tuple[bool, str]:
    """
    Returns (allowed, reason).
    - Block clearly harmful/unsafe intents (HARMFUL_BLOCKLIST, profanity)
    - Allow only finance-related queries (FINANCIAL_KEYWORDS). Everything else is 'irrelevant'.
    """
    q = query.lower().strip()

    # Harmful / unsafe content
    for pat in HARMFUL_BLOCKLIST:
        if re.search(pat, q):
            return (False, "harmful_disallowed")

    for pat in PROFANITY:
        if re.search(pat, q):
            return (False, "abusive_language")

    # Irrelevant if no financial keywords present
    if not any(kw in q for kw in FINANCIAL_KEYWORDS):
        return (False, "irrelevant")

    # Optional: overly long / nonsense input
    if len(q) > 1000:
        return (False, "too_long")

    return (True, "ok")


def guardrail_input_response(query: str, reason: str) -> str:
    if reason == "irrelevant":
        return ("Your query seems unrelated to the financial statements. "
                "Please ask about figures, sections, or disclosures in the company’s reports.")
    if reason == "harmful_disallowed":
        return ("This request is not allowed. I can’t help with harmful or unsafe instructions.")
    if reason == "abusive_language":
        return ("I can’t proceed due to abusive language. Please rephrase your question.")
    if reason == "too_long":
        return ("Your query is too long. Please shorten it and ask a specific question about the statements.")
    return ("I can’t proceed with this query. Please ask about the financial statements.")


# ---- Output guardrail ----

NUM_PATTERN = re.compile(r"(?:\$?\s?-?\d[\d,]*\.?\d*)")
CURRENCY_SYMBOLS = {"$", "₹", "€", "£"}

def _normalize_numbers(text: str) -> List[str]:
    """Extract numbers and normalize by removing commas/spaces; keep sign and decimals."""
    nums = []
    for m in NUM_PATTERN.finditer(text):
        raw = m.group(0)
        # Keep a version without currency/commas/spaces for matching
        cleaned = re.sub(r"[,\s$₹€£]", "", raw)
        if cleaned:
            nums.append(cleaned)
    return nums

def _contains_currency(text: str) -> bool:
    return any(sym in text for sym in CURRENCY_SYMBOLS)

def output_guardrail_verify(answer: str, context_docs: List[str]) -> Dict[str, str]:
    """
    Heuristic verifier to filter/flag hallucinated or non-factual outputs:
    - Every numeric value in answer must appear (normalized) in the combined context.
    - If answer has no numbers/currency and uses definitive tone, require decent overlap with context.
    Returns a dict with keys: status ('pass'|'flag'|'fail') and message (optional).
    """
    context = "\n".join(context_docs)
    context_norm = re.sub(r"[,\s$₹€£]", "", context)

    # 1) Numeric grounding check
    ans_nums = _normalize_numbers(answer)
    missing = [n for n in ans_nums if n not in context_norm]

    if missing:
        return {
            "status": "flag",
            "message": ("The answer includes numeric values not found in the retrieved context. "
                        "Potential hallucination detected.")
        }

    # 2) Non-numeric definitive claims: require minimum lexical overlap
    #    (very lightweight — avoids another model call)
    if not ans_nums and not _contains_currency(answer):
        ans_tokens = [t for t in re.findall(r"[a-zA-Z]{3,}", answer.lower())]
        ctx_tokens = set(re.findall(r"[a-zA-Z]{3,}", context.lower()))
        overlap = sum(1 for t in ans_tokens if t in ctx_tokens)
        if overlap < max(5, len(ans_tokens) // 4):  # require some overlap
            return {
                "status": "flag",
                "message": ("The answer is weakly grounded in the retrieved context. "
                            "It may be non-factual.")
            }

    return {"status": "pass", "message": "Answer grounded in retrieved context."}
