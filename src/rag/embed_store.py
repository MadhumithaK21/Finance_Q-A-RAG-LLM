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
    """
    Build and save a FAISS vector store from a PDF's chunked sections for retrieval.
    Args:
        pdf_path (str): Path to the PDF file.
        faiss_path (str): Where to save the FAISS index.
        embed_model (str): HuggingFace embedding model to use.
    """
    # 1. Chunk PDF into sections using custom logic
    chunks = chunk_sections(pdf_path)

    # 2. Ensure each chunk has page and chunk_size metadata for better retrieval and UI
    for c in chunks:
        if "page" not in c.metadata:
            c.metadata["page"] = c.metadata.get("source", "unknown")
        if "chunk_size" not in c.metadata:
            c.metadata["chunk_size"] = len(c.page_content.split())

    # 3. Build embeddings for each chunk using the specified model
    embeddings = HuggingFaceEmbeddings(
        model_name=embed_model,
        model_kwargs={"device": os.getenv("HF_DEVICE", "cpu")},
        encode_kwargs={"normalize_embeddings": True},
    )

    # 4. Build FAISS vector store from the chunked documents and their embeddings
    vector_store = FAISS.from_documents(chunks, embeddings)

    # 5. Save the FAISS index locally for later retrieval
    os.makedirs(os.path.dirname(faiss_path), exist_ok=True)
    vector_store.save_local(faiss_path)
    print(f"FAISS store saved to {faiss_path} using {embed_model}")


# Example usage: build FAISS index from a sample PDF
if __name__ == "__main__":
    pdf_path = os.path.join("data", "statements", "TSP-FS-Dec2021.pdf")
    build_faiss_store(pdf_path)
