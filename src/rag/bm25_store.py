import os
import pickle
from rank_bm25 import BM25Okapi
from chunk_sections import chunk_sections

def build_bm25_store(
    pdf_path,
    bm25_path="src/rag/vector_store/bm25.pkl",
):
    # 1. Chunk PDF
    chunks = chunk_sections(pdf_path)

    # 2. Add page & chunk_size metadata if missing
    for c in chunks:
        if "page" not in c.metadata:
            c.metadata["page"] = c.metadata.get("source", "unknown")
        if "chunk_size" not in c.metadata:
            c.metadata["chunk_size"] = len(c.page_content.split())

    # 3. Tokenize for BM25
    tokenized_corpus = [c.page_content.lower().split() for c in chunks]

    bm25 = BM25Okapi(tokenized_corpus)

    # 4. Save BM25 object and docs
    os.makedirs(os.path.dirname(bm25_path), exist_ok=True)
    with open(bm25_path, "wb") as f:
        pickle.dump({"bm25": bm25, "docs": chunks}, f)

    print(f"BM25 index saved to {bm25_path}")

if __name__ == "__main__":
    pdf_path = os.path.join("data", "statements", "TSP-FS-Dec2021.pdf")
    build_bm25_store(pdf_path)
