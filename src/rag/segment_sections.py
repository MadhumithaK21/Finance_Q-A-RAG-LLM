import os
import re
from ingest import load_and_clean
from langchain.schema import Document

SECTION_HEADERS = [
    r"independent auditors' report",
    r"statements of net assets available for benefits",
    r"statements of changes in net assets available for benefits",
    r"notes to financial statements",
    r"appendix 1",
    r"appendix 2"
]

def segment_tsp_sections(docs):
    sections = []
    current_title = None
    current_text = []

    for doc in docs:
        text_lower = doc.page_content.lower()
        found_header = None
        for pattern in SECTION_HEADERS:
            if re.search(pattern, text_lower):
                found_header = pattern
                break

        if found_header:
            # Save previous section
            if current_title and current_text:
                sections.append(Document(
                    page_content="\n".join(current_text),
                    metadata={"section": current_title}
                ))
                current_text = []

            current_title = found_header
            current_text.append(doc.page_content)
        else:
            if current_title:
                current_text.append(doc.page_content)

    # Save the last section
    if current_title and current_text:
        sections.append(Document(
            page_content="\n".join(current_text),
            metadata={"section": current_title}
        ))

    return sections

if __name__ == "__main__":
    pdf_path = os.path.join("data", "statements", "TSP-FS-Dec2021.pdf")
    docs = load_and_clean(pdf_path)
    sections = segment_tsp_sections(docs)
    print(f"Segmented into {len(sections)} sections.")
    for s in sections:
        print(f"--- {s.metadata['section']} ---")
        print(s.page_content[:200], "\n")
