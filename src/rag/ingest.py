import os
import re
import pdfplumber
from langchain.schema import Document


# Clean up extracted text by removing extra whitespace and page markers
def clean_text(text: str) -> str:
    """
    Basic cleanup for extracted text from PDF pages.
    - Removes multiple spaces/newlines
    - Removes page number markers
    """
    text = re.sub(r'\s+', ' ', text)  # remove multiple spaces/newlines
    text = re.sub(r'Page\s*\d+\s*(of\s*\d+)?', '', text, flags=re.IGNORECASE)
    return text.strip()


# Load a PDF, extract and clean text (including tables), and return as Document objects
def load_and_clean(pdf_path: str):
    """
    Extracts and cleans text (including tables) from each page of a PDF.
    Returns a list of Document objects, one per page with metadata.
    """
    docs = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            # Extract normal text from the page
            text = page.extract_text() or ""

            # Extract tables, if any, and append as readable text
            tables = page.extract_tables()
            for table in tables:
                # Convert table rows to a readable text format
                table_text = "\n".join(
                    [" | ".join(cell if cell else "" for cell in row) for row in table]
                )
                text += "\n" + table_text

            # Clean and store if there's any content on the page
            if text.strip():
                cleaned = clean_text(text)
                docs.append(Document(page_content=cleaned, metadata={"page": i + 1}))
    return docs


# load and clean a sample PDF, then print info about the first page
if __name__ == "__main__":
    pdf_path = os.path.join("data", "statements", "TSP-FS-Dec2021.pdf")
    docs = load_and_clean(pdf_path)
    print(f"Loaded {len(docs)} pages after cleaning.")
    print(docs[0].page_content[:500])
