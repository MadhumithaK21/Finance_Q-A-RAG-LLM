import os
import pickle
import re
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from rank_bm25 import BM25Okapi

# Utilities
from src.rag.query_utils import preprocess_query
from src.rag.guardrails import (
    input_guardrail_check,
    guardrail_input_response,
    output_guardrail_verify,
)

load_dotenv()


def _build_hf_embeddings():
    """
    Use an open-source sentence embedding model suitable for retrieval.
    E5 models work very well (remember: E5 expects 'query: ' / 'passage: ' during encoding).
    Ensure ingest/embedding uses the same model to build FAISS.
    """
    model_name = os.getenv("EMBED_MODEL", "intfloat/e5-small-v2")
    emb = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": os.getenv("HF_DEVICE", "cpu")},
        encode_kwargs={"normalize_embeddings": True},
    )
    return emb


def _build_hf_llm():
    """
    Use a small open-source seq2seq model (FLAN-T5) for generation.
    Prefer your local fine-tuned folder if present: models/flan-t5-financial
    """
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

    local_ft_path = "models/flan-t5-financial"  # your path
    model_name = (
        local_ft_path if os.path.isdir(local_ft_path)
        else os.getenv("HF_RAG_MODEL", "google/flan-t5-small")
    )

    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    gen_pipe = pipeline(
        task="text2text-generation",
        model=mdl,
        tokenizer=tok,
        max_new_tokens=int(os.getenv("HF_MAX_NEW_TOKENS", "64")),
        temperature=float(os.getenv("HF_TEMPERATURE", "0.0")),
        do_sample=False,
    )
    return HuggingFacePipeline(pipeline=gen_pipe)


class HybridQA:
    """Hybrid (FAISS + BM25) retrieval with input/output guardrails and dual-chunk mixing."""

    def __init__(self, llm, faiss_retriever, bm25, bm25_docs, prompt):
        self.llm = llm
        self.faiss_retriever = faiss_retriever  # should be a LangChain retriever (Runnable)
        self.bm25 = bm25                       # rank_bm25.BM25Okapi
        self.bm25_docs = bm25_docs             # list[Document]
        self.prompt = prompt                   # PromptTemplate

    def bm25_retrieve(self, query, k=5):
        processed = preprocess_query(query)
        tokens = processed.split()
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(zip(scores, self.bm25_docs), key=lambda x: x[0], reverse=True)[:k]
        return [doc for _, doc in ranked]

    def hybrid_retrieve(self, query, k=6):
        processed = preprocess_query(query)

        # Dense retrieval (FAISS)
        # New-style retrievers in LangChain are Runnable: .invoke(query_string)
        dense_results = []
        try:
            dense_results = self.faiss_retriever.invoke(processed)
        except Exception:
            # fallback to legacy API
            try:
                dense_results = self.faiss_retriever.get_relevant_documents(processed)
            except Exception:
                dense_results = []

        # Mix short + long chunks explicitly (expects metadata["chunk_size"] in {500, 1000})
        short_chunks = [d for d in dense_results if d.metadata.get("chunk_size") == 500]
        long_chunks = [d for d in dense_results if d.metadata.get("chunk_size") == 1000]
        short_top = short_chunks[: max(1, k // 2)]
        long_top = long_chunks[: max(1, k - len(short_top))]

        # Sparse retrieval (BM25)
        sparse_results = self.bm25_retrieve(processed, k=k)

        # Combine and dedupe (prefer a stable key if you embedded one, e.g., metadata["id"])
        combined = {}
        def key(doc):
            return doc.metadata.get("id") or doc.page_content
        for d in short_top + long_top + sparse_results:
            combined[key(d)] = d

        return list(combined.values())

    def invoke(self, query: str, k=6):
        # ---- INPUT GUARDRAIL ----
        allowed, reason = input_guardrail_check(query)
        if not allowed:
            return {"result": guardrail_input_response(query, reason), "source_documents": []}

        # ---- RETRIEVE ----
        docs = self.hybrid_retrieve(query, k=k)
        context_texts = [d.page_content for d in docs]
        context = "\n\n".join(context_texts)

        # ---- GENERATE ----
        prompt_text = self.prompt.format(context=context, question=query)
        try:
            answer = self.llm.invoke(prompt_text)  # HuggingFacePipeline supports .invoke
        except Exception:
            # Fallback to call-style interface if needed
            answer = self.llm(prompt_text)

        # Coerce to plain string
        answer_str = str(getattr(answer, "content", answer)).strip()

        # ---- OUTPUT GUARDRAIL ----
        verification = output_guardrail_verify(answer_str, context_texts)
        status = verification.get("status")

        if status == "pass":
            return {"result": answer_str, "source_documents": docs}

        if status in ("flag", "fail"):
            safe_msg = (
                "I could not find that in the provided financial statements."
            )
            return {"result": safe_msg, "source_documents": docs}

        # Default conservative
        return {"result": "I could not find that in the provided financial statements.", "source_documents": docs}


def get_rag_qa(
    faiss_path="src/rag/vector_store/faiss_index",   # REBUILD with E5-small-v2
    bm25_path="src/rag/vector_store/bm25.pkl",
    hybrid=True,
):
    # ---- FAISS retriever ----
    embeddings = _build_hf_embeddings()
    vector_store = FAISS.load_local(
        faiss_path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )
    faiss_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6},
    )

    # ---- BM25 retriever ----
    with open(bm25_path, "rb") as f:
        bm25_data = pickle.load(f)
    bm25 = bm25_data["bm25"]               # BM25Okapi
    bm25_docs = bm25_data["docs"]          # list[Document]

    # ---- Prompt & LLM (HF) ----
    template = """
You are a financial QA assistant answering from official company financial statements.

You will be given extracted text from the statements.
Rules:
- Return ONLY the exact value(s) from the context (keep currency symbols/units/percents as shown).
- Do NOT add any explanation.
- If multiple candidates exist, pick the most relevant by context and year.
- If not found in the context, output exactly: I could not find that in the provided financial statements.

Context:
{context}

Question: {question}

Answer:
"""
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    llm = _build_hf_llm()

    if hybrid:
        chain = HybridQA(llm, faiss_retriever, bm25, bm25_docs, prompt)
        retriever = chain.hybrid_retrieve
    else:
        # FAISS-only fallback (kept for completeness)
        from langchain.chains import RetrievalQA
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=faiss_retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True,
        )
        retriever = faiss_retriever

    return chain, retriever


if __name__ == "__main__":
    qa, retriever = get_rag_qa(hybrid=True)
    test_queries = [
        "What were participant contributions in 2020?",
        "What was due for securities purchased in 2021?",
        "Show cash collateral payable in 2020",
        "Who is the CEO?",                      # irrelevant
        "how to hack bank systems",             # harmful → should be blocked
    ]

    for q in test_queries:
        print(f"\n=== Query: {q} ===")
        try:
            docs = retriever(q, k=6) if callable(retriever) else retriever.get_relevant_documents(q)
        except Exception:
            docs = []
        for i, d in enumerate(docs[:3], start=1):
            print(f"  - Doc{i} section={d.metadata.get('section')} size={d.metadata.get('chunk_size')}")
        out = qa.invoke(q)
        print("Answer:", out["result"])
