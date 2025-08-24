import os
import time
import csv
import numpy as np
import re
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate

load_dotenv()

FAISS_PATH = "vector_store/faiss_index"
LOG_FILE = "query_logs.csv"

# ----------------------------
# Build RAG QA components
# ----------------------------
def get_rag_qa():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vector_store = FAISS.load_local(
        FAISS_PATH,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )

    template = """
    You are a financial QA assistant specializing in answering questions from official company financial statements.

    You will be given extracted text from the statements.
    The text may include numbers with commas, dollar signs, percentages, or spaces between digits.
    When answering:
    - Return ONLY the exact number(s) or value(s) found in the context, with currency symbols or units if present.
    - Do NOT add any explanation or extra words.
    - If multiple values match, return the most relevant one for the question.
    - If you cannot find the answer in the context, return exactly: "Not found".

    Context:
    {context}

    Question: {question}

    Answer:
    """

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    return llm, vector_store, prompt


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Financial QA - RAG", layout="wide")
st.title("Financial Statements Q&A")

method = st.radio("Select method:", ["RAG", "Fine-Tune"], horizontal=True)

llm, vector_store, prompt = get_rag_qa()

query = st.text_input("Ask a question about the financial statements:")

if query:
    start_time = time.time()
    with st.spinner(f"Running {method} mode..."):

        if method == "RAG":
            # Step 1: Initial retrieval
            retrieved_docs_and_scores = vector_store.similarity_search_with_score(query, k=8)

            # Step 2: Re-rank using cosine similarity
            query_embedding = vector_store.embedding_function.embed_query(query)
            reranked = []
            for doc, _ in retrieved_docs_and_scores:
                doc_embedding = vector_store.embedding_function.embed_query(doc.page_content)
                sim = np.dot(query_embedding, doc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                )
                reranked.append((doc, sim))

            reranked.sort(key=lambda x: x[1], reverse=True)
            retrieved_docs = [doc for doc, _ in reranked[:2]]
            cosine_sims = [sim for _, sim in reranked[:2]]
            avg_confidence = sum(cosine_sims) / len(cosine_sims) if cosine_sims else 0

            # Step 3: Prepare context
            context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

            # Step 4: Run LLM with context
            final_prompt = prompt.format(context=context_text, question=query)
            result = llm.invoke(final_prompt)
            answer = result.content.strip()

            # Step 5: Post-process to remove extra spaces in numbers
            answer = re.sub(r"(?<=\d)\s+(?=\d)", "", answer)
            answer = re.sub(r",+", ",", answer)

        elif method == "Fine-Tune":
            avg_confidence = 0
            answer = "FT mode not implemented yet"

        end_time = time.time()

        # Display results
        st.subheader("Answer")
        st.write(answer)
        st.markdown(
            f"**Retrieval Confidence Score:** "
            f"<span style='color:green; font-weight:bold'>{avg_confidence:.4f}</span>",
            unsafe_allow_html=True
        )
        st.markdown(f"**Method Used:** {method}")
        st.markdown(f"**Response Time:** {end_time - start_time:.2f} seconds")

        if method == "RAG":
            st.subheader("Sources")
            for i, doc in enumerate(retrieved_docs, start=1):
                page_num = doc.metadata.get("page", "Unknown")
                st.markdown(f"**Source {i}** - Page {page_num}")
                st.caption(doc.page_content[:500] + "...")

        # Log to CSV
        log_data = [
            query,
            method,
            answer,
            f"{avg_confidence:.4f}",
            f"{end_time - start_time:.2f}",
            ""  # Correct (Y/N) placeholder
        ]
        try:
            with open(LOG_FILE, mode="x", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Question", "Method", "Answer", "Confidence", "Time (s)", "Correct (Y/N)"])
                writer.writerow(log_data)
        except FileExistsError:
            with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(log_data)
