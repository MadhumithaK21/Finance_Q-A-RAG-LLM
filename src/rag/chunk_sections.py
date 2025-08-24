import os
import uuid
from ingest import load_and_clean
from segment_sections import segment_tsp_sections
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

def chunk_sections(pdf_path, chunk_sizes=[500, 1000], chunk_overlap=200):
    """
    Split the PDF into multiple chunk sizes for dual indexing.
    Each chunk stores metadata about its section, size, and unique ID.
    """
    docs = load_and_clean(pdf_path)
    sections = segment_tsp_sections(docs)

    all_chunks = []

    for size in chunk_sizes:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=chunk_overlap,
        )

        for sec in sections:
            chunks = splitter.split_documents([sec])
            for chunk in chunks:
                chunk.metadata["section"] = sec.metadata["section"]
                chunk.metadata["chunk_size"] = size
                chunk.metadata["id"] = str(uuid.uuid4())
                all_chunks.append(chunk)

    return all_chunks


if __name__ == "__main__":
    pdf_path = os.path.join("data", "statements", "TSP-FS-Dec2021.pdf")
    chunks = chunk_sections(pdf_path)
    print(f"Created {len(chunks)} chunks (across sizes).")
    print(chunks[0].metadata)
    print(chunks[0].page_content[:300])
