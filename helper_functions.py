from langchain.schema import BaseRetriever
from typing import List
from langchain.schema import Document

class StaticRetriever(BaseRetriever):
    docs: List[Document]
    k: int = 5

    # This get_relevant_documents method is required by the BaseRetriever
    def _get_relevant_documents(self, query: str) -> List[Document]:
        return self.docs[:self.k]

    # This is not strictly required but good practice for async operations
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self.docs[:self.k]



from sentence_transformers import CrossEncoder

def cross_encoder_rerank(query, documents, top_k:int):
    """Rerank using cross-encoder model"""
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    # Score query-document pairs
    doc_texts = [doc.page_content for doc in documents]
    scores = cross_encoder.predict([(query, doc) for doc in doc_texts])
    
    # Sort by scores and return top documents
    scored_docs = list(zip(scores, documents))
    scored_docs.sort(reverse=True, key=lambda x: x[0])
    
    return [doc for _, doc in scored_docs[:top_k]]



from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os


load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in .env file or environment variables.")

llm = ChatGroq(groq_api_key=groq_api_key,
               model_name="llama-3.1-8b-instant",
               max_tokens = 4096)