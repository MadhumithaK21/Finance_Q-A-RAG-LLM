
# Build and save a BM25 index from a PDF's chunked sections for retrieval
import os
import pickle
from rank_bm25 import BM25Okapi
from chunk_sections import chunk_sections


def build_bm25_store(
    pdf_path,
    bm25_path="src/rag/vector_store/bm25.pkl",
):
    # 1. Chunk the PDF into sections using custom logic
    chunks = chunk_sections(pdf_path)

    # 2. Ensure each chunk has page and chunk_size metadata for better retrieval and UI
    for c in chunks:
        if "page" not in c.metadata:
            c.metadata["page"] = c.metadata.get("source", "unknown")
        if "chunk_size" not in c.metadata:
            c.metadata["chunk_size"] = len(c.page_content.split())

    # 3. Tokenize each chunk's content for BM25 (lowercase, whitespace split)
    tokenized_corpus = [c.page_content.lower().split() for c in chunks]

    # 4. Build the BM25 index from the tokenized corpus
    bm25 = BM25Okapi(tokenized_corpus)

    # 5. Save the BM25 object and the original chunked documents for later retrieval
    os.makedirs(os.path.dirname(bm25_path), exist_ok=True)
    with open(bm25_path, "wb") as f:
        pickle.dump({"bm25": bm25, "docs": chunks}, f)

    print(f"BM25 index saved to {bm25_path}")


# Example usage: build BM25 index from a sample PDF
if __name__ == "__main__":
    pdf_path = os.path.join("data", "statements", "TSP-FS-Dec2021.pdf")
    build_bm25_store(pdf_path)
