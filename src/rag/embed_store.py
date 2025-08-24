import os
import pickle
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from chunk_sections import chunk_sections

load_dotenv()

def build_faiss_store(
    pdf_path,
    faiss_path="src/rag/vector_store/faiss_index",
    embed_model="intfloat/e5-small-v2",
):
    # 1. Chunk PDF into sections (your existing logic)
    chunks = chunk_sections(pdf_path)

    # 2. Add metadata (page, chunk_size) if missing — improves retrieval + UI
    for c in chunks:
        if "page" not in c.metadata:
            c.metadata["page"] = c.metadata.get("source", "unknown")
        if "chunk_size" not in c.metadata:
            c.metadata["chunk_size"] = len(c.page_content.split())

    # 3. Build embeddings (open-source E5 model)
    embeddings = HuggingFaceEmbeddings(
        model_name=embed_model,
        model_kwargs={"device": os.getenv("HF_DEVICE", "cpu")},
        encode_kwargs={"normalize_embeddings": True},
    )

    # 4. Build FAISS vector store
    vector_store = FAISS.from_documents(chunks, embeddings)

    # 5. Save locally
    os.makedirs(os.path.dirname(faiss_path), exist_ok=True)
    vector_store.save_local(faiss_path)
    print(f"FAISS store saved to {faiss_path} using {embed_model}")

if __name__ == "__main__":
    pdf_path = os.path.join("data", "statements", "TSP-FS-Dec2021.pdf")
    build_faiss_store(pdf_path)
