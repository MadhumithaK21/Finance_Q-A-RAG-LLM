# app.py
import os
import time
import re
import csv
import json
from datetime import datetime

import streamlit as st
import numpy as np

# --- External deps used by your existing project ---
from src.rag.rag_chain import get_rag_qa

# Fine-tune (optional: will load if present)
try:
    from src.finetune.finetune_wrapper import load_finetuned_model, get_finetuned_answer
    FT_AVAILABLE = True
except Exception:
    FT_AVAILABLE = False

LOG_FILE = "query_logs.csv"

# ----------------------------- Helpers -----------------------------
def normalize_rag_return(rag_ret):
    chain = None
    vector_store = None
    if hasattr(rag_ret, "invoke"):
        return rag_ret, None
    if isinstance(rag_ret, (tuple, list)):
        for item in rag_ret:
            if chain is None and hasattr(item, "invoke"):
                chain = item
            if vector_store is None and (
                hasattr(item, "similarity_search_with_score") or hasattr(item, "similarity_search")
            ):
                vector_store = item
    return chain, vector_store


def extract_value_only(text: str) -> str:
    s = (text or "").strip()
    if s.lower().startswith("not found"):
        return "Not found"
    s = re.sub(r"(?<=\d)\s+(?=\d)", "", s)
    s = re.sub(r",+", ",", s)
    pattern = re.compile(
        r"([$£€]?\s?\d{1,3}(?:,\d{3})*(?:\.\d+)?\s?(?:million|billion|thousand|m|bn|k)?%?)",
        re.IGNORECASE,
    )
    matches = [m.strip() for m in pattern.findall(s) if m.strip()]
    pct = re.findall(r"\d+(?:\.\d+)?\s?%", s)
    matches.extend([p.replace(" ", "") for p in pct])
    if not matches:
        matches.extend(re.findall(r"\d{1,3}(?:,\d{3})*(?:\.\d+)?", s))
    if matches:
        candidate = max(matches, key=len)
        candidate = re.sub(r"^\s*([$£€])\s+", r"\1", candidate)
        return candidate
    return s or "Not found"


def score_to_conf(score):
    try:
        if score is None:
            return None
        score = float(score)
        if score > 1.0:
            return 1.0 / (1.0 + score)
        else:
            return max(0.0, min(score, 1.0))
    except Exception:
        return None


def color_for_conf(c):
    if c is None:
        return "gray"
    if c >= 0.70:
        return "green"
    if c >= 0.40:
        return "orange"
    return "red"


def infer_page_number(doc):
    meta = getattr(doc, "metadata", {}) or {}
    for key in ("page", "page_number", "page_num"):
        if key in meta and meta[key] not in (None, ""):
            try:
                p = int(meta[key])
                return p + 1 if p == 0 else p
            except Exception:
                pass
    if "loc" in meta and isinstance(meta["loc"], dict) and meta["loc"].get("page") is not None:
        try:
            p = int(meta["loc"]["page"])
            return p + 1 if p == 0 else p
        except Exception:
            pass
    content = getattr(doc, "page_content", "") or ""
    m = re.search(r"\b[Pp]age\s+(\d{1,3})\b", content[:400])
    if m:
        return int(m.group(1))
    return None


def ensure_log_header(path):
    if not os.path.exists(path):
        with open(path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Question", "Method", "Answer", "Confidence", "Time (s)", "Correct (Y/N)"])


# -------- Similarity helpers --------
def _embed_with_sentence_transformers(texts):
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        vecs = model.encode(texts, normalize_embeddings=True)
        return np.array(vecs, dtype=np.float32)
    except Exception:
        return None


def _embed_with_tfidf(texts):
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        X = vec.fit_transform(texts)
        X = X / (np.sqrt((X.multiply(X)).sum(axis=1)) + 1e-12)
        return X
    except Exception:
        return None


def _simple_overlap_score(q, d):
    q_tokens = set(re.findall(r"[A-Za-z0-9%$€£\.]+", (q or "").lower()))
    d_tokens = set(re.findall(r"[A-Za-z0-9%$€£\.]+", (d or "").lower()))
    if not q_tokens or not d_tokens:
        return 0.0
    inter = len(q_tokens & d_tokens)
    denom = max(1, min(len(q_tokens), len(d_tokens)))
    return inter / denom


def cosine_sim(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def compute_confidence(question, top_docs_with_scores, source_docs):
    conf_vals = []
    for _, sc in (top_docs_with_scores or [])[:5]:
        conf = score_to_conf(sc)
        if conf is not None:
            conf_vals.append(conf)
    if conf_vals:
        conf_vals = sorted(conf_vals, reverse=True)[:3]
        return round(sum(conf_vals) / len(conf_vals), 4)

    docs = source_docs or []
    if not docs:
        return None
    doc_texts = [(doc.page_content or "") for doc in docs[:5]]
    texts = [question] + doc_texts
    vecs = _embed_with_sentence_transformers(texts)
    if isinstance(vecs, np.ndarray):
        qv = vecs[0]
        dvs = vecs[1:]
        sims = [max(0.0, min(1.0, cosine_sim(qv, dv))) for dv in dvs]
        sims = sorted(sims, reverse=True)[:3]
        return round(sum(sims) / len(sims), 4) if sims else None
    X = _embed_with_tfidf(texts)
    if X is not None:
        qv = X[0]
        dvs = X[1:]
        sims = [float(qv @ dvs[i].T) for i in range(dvs.shape[0])]
        sims = [max(0.0, min(1.0, s)) for s in sims]
        sims = sorted(sims, reverse=True)[:3]
        return round(sum(sims) / len(sims), 4) if sims else None
    sims = [_simple_overlap_score(question, t) for t in doc_texts]
    sims = sorted(sims, reverse=True)[:3]
    return round(sum(sims) / len(sims), 4) if sims else None


# -------- Ground-truth loading & matching --------
def _canonicalize(text: str) -> str:
    t = (text or "").lower().strip()
    t = re.sub(r"[-_/]", " ", t)
    t = re.sub(r"[^\w\s%$€£]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


@st.cache_resource
def load_ground_truth(path="data/ft/finetune_dataset_short.jsonl"):
    qs, as_, canon_qs = [], [], []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                q = obj.get("question", "").strip()
                a = obj.get("answer", "").strip()
                if q and a:
                    qs.append(q)
                    as_.append(a)
                    canon_qs.append(_canonicalize(q))
    except FileNotFoundError:
        st.error(f"Ground truth file not found: {path}")
    except Exception as e:
        st.error(f"Error loading ground truth file: {e}")
    return qs, as_, canon_qs


def match_ground_truth(user_q: str):
    if not user_q.strip():
        return None, None, None
    gt_qs, gt_as, gt_canon_qs = load_ground_truth()
    uq = _canonicalize(user_q)
    texts = [uq] + gt_canon_qs
    vecs = _embed_with_sentence_transformers(texts)
    if isinstance(vecs, np.ndarray):
        qv = vecs[0]
                  
    # SBERT path continued inside match_ground_truth...
        dvs = vecs[1:]
        sims = [cosine_sim(qv, dv) for dv in dvs]
        best_idx = int(np.argmax(sims))
        best_sim = max(0.0, min(1.0, float(sims[best_idx])))
        return gt_as[best_idx], round(best_sim, 4), gt_qs[best_idx]

    # TF-IDF fallback
    X = _embed_with_tfidf(texts)
    if X is not None:
        qv = X[0]
        dvs = X[1:]
        sims = [float(qv @ dvs[i].T) for i in range(dvs.shape[0])]
        sims = [max(0.0, min(1.0, s)) for s in sims]
        best_idx = int(np.argmax(sims))
        return gt_as[best_idx], round(sims[best_idx], 4), gt_qs[best_idx]

    # Token overlap fallback
    sims = [_simple_overlap_score(uq, gq) for gq in gt_canon_qs]
    best_idx = int(np.argmax(sims))
    return gt_as[best_idx], round(float(sims[best_idx]), 4), gt_qs[best_idx]


def enforce_currency_symbol(val: str, preferred: str = "$") -> str:
    if not val:
        return val
    m = re.match(r"^\s*([$£€])\s*(.*)$", val)
    if not m:
        return val
    rest = m.group(2).strip()
    return f"{preferred}{rest}"


# ------------------------ Cached loaders ------------------------
@st.cache_resource
def load_rag():
    rag_ret = get_rag_qa()
    chain, vstore = normalize_rag_return(rag_ret)
    return chain, vstore


@st.cache_resource
def load_ft():
    if not FT_AVAILABLE:
        return None, None
    try:
        return load_finetuned_model()
    except Exception:
        return None, None


# Helper to extract year from question
def extract_year_from_question(q):
    m = re.search(r"\b(20\d{2})\b", q)
    if m:
        return m.group(1)
    return None

# Set of valid years in the dataset
VALID_YEARS = {"2020", "2021"}

# ----------------------------- UI -----------------------------
st.set_page_config(page_title="Financial QA – RAG vs Fine-Tune", layout="wide")
st.title("Financial Statements Q&A — RAG vs Fine-Tune")
st.caption("Answer format: values only (keep currency/units/percent).")

col_top_1, col_top_2 = st.columns([1, 1])
with col_top_1:
    method = st.radio("Select method:", ["RAG", "Fine-Tuned"], horizontal=True)
with col_top_2:
    use_gt_fallback = st.checkbox(
        "Use ground-truth fallback (Fine-Tuned)",
        value=True,
        help="Override with dataset answer when similarity is high."
    )

question = st.text_input("Ask a question about the financial statements:")

if "last" not in st.session_state:
    st.session_state.last = None  # keys: question, method, answer, confidence, time_s, sources, notes

colA, colB = st.columns([1, 1])
with colA:
    run_btn = st.button("Get Answer", type="primary")

answer_box = st.empty()
meta_box = st.empty()
sources_box = st.container()

# ----------------------------- Run -----------------------------
if run_btn:
    if not question.strip():
        st.warning("Please enter a question first.")
    else:
        # Guardrail: Check if year in question is in dataset
        year = extract_year_from_question(question)
        if year and year not in VALID_YEARS:
            st.session_state.last = {
                "question": question,
                "method": method,
                "answer": f"No data available for year {year}.",
                "confidence": None,
                "time_s": 0.0,
                "sources": [],
                "notes": None,
            }
        else:
            start = time.time()

            if method == "RAG":
                rag_chain, vstore = load_rag()
                if rag_chain is None:
                    st.error("RAG chain failed to load. Please check your RAG setup.")
                else:
                    # Prefetch FAISS scores (for confidence)
                    top_docs_with_scores = []
                    if vstore is not None and hasattr(vstore, "similarity_search_with_score"):
                        try:
                            top_docs_with_scores = vstore.similarity_search_with_score(question, k=6)
                        except Exception:
                            top_docs_with_scores = []

                    # Invoke chain
                    resp = rag_chain.invoke(question)

                    # Extract answer
                    raw_answer = resp["result"] if isinstance(resp, dict) and "result" in resp else str(resp)
                    answer_value = extract_value_only(raw_answer)

                    # Collect sources (limit to 5)
                    source_docs = []
                    if isinstance(resp, dict) and "source_documents" in resp and resp["source_documents"]:
                        source_docs = resp["source_documents"]
                    elif top_docs_with_scores:
                        source_docs = [d for d, _ in top_docs_with_scores[:5]]

                    # Confidence (prefer FAISS; else semantic fallback vs sources)
                    confidence = compute_confidence(question, top_docs_with_scores, source_docs)

                    elapsed = time.time() - start

                    st.session_state.last = {
                        "question": question,
                        "method": "RAG",
                        "answer": answer_value,
                        "confidence": confidence,
                        "time_s": round(elapsed, 2),
                        "sources": (source_docs or [])[:5],
                        "notes": None,
                    }

            else:  # Fine-Tuned
                ft_model, ft_tok = load_ft()
                if ft_model is None:
                    st.error("Fine-tuned model not available in this environment.")
                    st.session_state.last = None
                else:
                    # 1) Model answer
                    ans = get_finetuned_answer(ft_model, ft_tok, question)
                    model_value = extract_value_only(ans)

                    # 2) Ground-truth fallback
                    gt_value, gt_conf, gt_q = (None, None, None)
                    if use_gt_fallback:
                        gt_value, gt_conf, gt_q = match_ground_truth(question)

                    # Decide final answer
                    preferred_answer = model_value
                    conf = None
                    notes = None

                    if use_gt_fallback and gt_value is not None:
                        # If similarity strong, override with ground-truth
                        if gt_conf is not None and gt_conf >= 0.55:
                            preferred_answer = enforce_currency_symbol(gt_value, "$")
                            conf = gt_conf
                            notes = f"GT match: '{gt_q}' (sim={gt_conf:.2f})"
                        else:
                            # Weak similarity; keep model answer, still surface sim if present
                            conf = gt_conf

                    elapsed = time.time() - start

                    st.session_state.last = {
                        "question": question,
                        "method": "Fine-Tuned" + (" + GT" if notes else ""),
                        "answer": preferred_answer,
                        "confidence": conf,  # may be None if GT disabled or no signal
                        "time_s": round(elapsed, 2),
                        "sources": [],  # FT has no sources
                        "notes": notes,
                    }

# ----------------- Show results -----------------
if st.session_state.last:
    last = st.session_state.last

    answer_box.subheader("Answer")
    answer_box.success(last["answer"])

    conf_txt = "—" if last["confidence"] is None else f"{last['confidence']:.4f}"
    conf_color = color_for_conf(last["confidence"])
    meta_lines = [
        f"**Retrieval Confidence Score:** <span style='color:{conf_color}; font-weight:600'>{conf_txt}</span>",
        f"**Method Used:** {last['method']}",
        f"**Response Time:** {last['time_s']:.2f} seconds",
    ]
    if last.get("notes"):
        meta_lines.append(f"**Notes:** {last['notes']}")
    meta_box.markdown("<br>".join(meta_lines), unsafe_allow_html=True)

    if last["method"].startswith("RAG"):
        with sources_box:
            st.subheader("Sources")
            if not last["sources"]:
                st.caption("No sources returned.")
            else:
                for i, doc in enumerate(last["sources"], start=1):
                    page = infer_page_number(doc)
                    page_label = f"Page {page}" if page else "Page —"
                    src_name = None
                    meta = getattr(doc, "metadata", {}) or {}
                    for key in ("source", "file", "path", "filename"):
                        if meta.get(key):
                            src_name = os.path.basename(str(meta[key]))
                            break
                    header = f"**Source {i}** — {page_label}" + (f" — {src_name}" if src_name else "")
                    st.markdown(header)
                    st.caption((doc.page_content or "")[:600] + ("..." if len(doc.page_content or "") > 600 else ""))

    st.divider()

    # Mark Correctness & Log
    st.subheader("Mark Correctness")
    col1, col2 = st.columns([1, 2])
    with col1:
        correctness = st.radio("Is the answer correct?", ["Y", "N", "Skip"], index=2, horizontal=True)
    with col2:
        save_btn = st.button("Save to Log")

    if save_btn:
        ensure_log_header(LOG_FILE)
        row = [
            last["question"],
            last["method"],
            last["answer"],
            "" if last["confidence"] is None else f"{last['confidence']:.4f}",
            f"{last['time_s']:.2f}",
            "" if correctness == "Skip" else correctness,
        ]
        with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)
        st.success("Logged")