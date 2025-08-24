import numpy as np
import pickle
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

class HybridRetriever:
    def __init__(self, faiss_path, bm25_path, embedding_model=None):
        # Load FAISS
        self.embeddings = embedding_model or HuggingFaceEmbeddings(model_name="intfloat/e5-small-v2")
        self.vector_store = FAISS.load_local(
            faiss_path,
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True,
        )

        # Load BM25
        with open(bm25_path, "rb") as f:
            bm25_data = pickle.load(f)
        self.bm25 = bm25_data["bm25"]
        self.bm25_docs = bm25_data["docs"]

    def score_to_similarity(self, score):
        return 1.0 / (1.0 + float(score)) if score is not None else 0.0

    def normalize_scores(self, scores):
        min_score, max_score = min(scores), max(scores)
        if max_score == min_score:
            return [1.0] * len(scores)
        return [(s - min_score) / (max_score - min_score) for s in scores]

    def get_top_documents(self, query, k=8, top_n=2, fusion_weights=(0.6, 0.4)):
        # FAISS/Knowledge Base/Vector Store retrieval
        faiss_docs_and_scores = self.vector_store.similarity_search_with_score(query, k=k)
        faiss_docs = [doc for doc, _ in faiss_docs_and_scores]
        faiss_scores = [self.score_to_similarity(score) for _, score in faiss_docs_and_scores]

        # BM25 retrieval
        bm25_raw_scores = self.bm25.get_scores(query)
        bm25_top_indices = np.argsort(bm25_raw_scores)[::-1][:k]
        bm25_docs = [self.bm25_docs[i] for i in bm25_top_indices]
        bm25_scores = [bm25_raw_scores[i] for i in bm25_top_indices]
        bm25_norm_scores = self.normalize_scores(bm25_scores)

        # Combine and deduplicate
        all_docs = {}
        for doc, score in zip(faiss_docs, faiss_scores):
            all_docs[doc.page_content] = {"doc": doc, "faiss": score, "bm25": 0.0}
        for doc, score in zip(bm25_docs, bm25_norm_scores):
            if doc.page_content in all_docs:
                all_docs[doc.page_content]["bm25"] = score
            else:
                all_docs[doc.page_content] = {"doc": doc, "faiss": 0.0, "bm25": score}

        # Fuse scores
        fused_docs = []
        for entry in all_docs.values():
            fused_score = fusion_weights[0] * entry["faiss"] + fusion_weights[1] * entry["bm25"]
            fused_docs.append((entry["doc"], fused_score))

        # Cosine re-ranking
        query_embedding = self.embeddings.embed_query(query)
        reranked = []
        for doc, _ in fused_docs:
            doc_embedding = self.embeddings.embed_query(doc.page_content)
            sim = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            reranked.append((doc, sim))

        reranked.sort(key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, _ in reranked[:top_n]]
        top_sims = [sim for _, sim in reranked[:top_n]]
        avg_confidence = sum(top_sims) / len(top_sims) if top_sims else 0

        return top_docs, avg_confidence