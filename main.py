import pandas as pd
import os
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from proposal import proposal
load_dotenv()


groq_api_key = os.getenv("GROQ_API_KEY")


if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in .env file or environment variables.")

embedding_model = HuggingFaceEmbeddings(model_name = "models/local_all-MiniLM-L6-v2")

vectorstore = Chroma(   
    persist_directory="./chromaDB",
    embedding_function=embedding_model,
    collection_name="proposal_collection"  # same name as before
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})


vectorstore_guidelines = Chroma(   
    persist_directory="./GuidelinesDB",
    embedding_function=embedding_model,
    collection_name="guidelines_collection"  # same name as before
)
retriever_guidelines = vectorstore_guidelines.as_retriever()

llm = ChatGroq(groq_api_key=groq_api_key,
               model_name="llama-3.1-8b-instant",
               max_tokens = 4096)


rag_chain = RetrievalQA.from_chain_type(llm=llm, 
                                        retriever=retriever_guidelines,
                                        chain_type="map_reduce", 
                                        return_source_documents=True)  # Optional: useful for debugging

query = f"""
    my proposal is {proposal}
    You are an expert reviewer working for the Nodal Agency (CMPDI) under the Ministry of Coal. Your task is to conduct a meticulous and thorough compliance audit of the provided research proposal against the official 'GUIDELINES FOR RESEARCH PROJECTS OF MINISTRY OF COAL.
    
"""

# response = rag_chain.run(query)
# print("🔎 Answer:", response)


response = rag_chain.invoke({"query": query})



print(response["result"])  # The answer text
print('--------------------------------------------------------')
print(response["source_documents"]) # List of source documents
print(len(response["source_documents"]))