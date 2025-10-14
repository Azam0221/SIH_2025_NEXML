import pandas as pd
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from proposal import proposal
from helper_functions import llm, cross_encoder_rerank, StaticRetriever

embedding_model = HuggingFaceEmbeddings(model_name = "models/local_all-MiniLM-L6-v2")

vectorstore_guidelines = Chroma(   
    persist_directory="./GuidelinesDB",
    embedding_function=embedding_model,
    collection_name="guidelines_collection"  # same name as before
)
retriever_guidelines = vectorstore_guidelines.as_retriever(search_kwargs={"k": 15})

print("Starting to Retreive to 15 Guidelines")


rag_chain = RetrievalQA.from_chain_type(llm=llm, 
                                        retriever=retriever_guidelines,
                                        chain_type="stuff",  
                                        return_source_documents=True)  
query = f"""
    my proposal is {proposal}
    You are an expert reviewer working for the Nodal Agency (CMPDI) under the Ministry of Coal. Your task is to conduct a meticulous and thorough compliance audit of the provided research proposal against the official 'GUIDELINES FOR RESEARCH PROJECTS OF MINISTRY OF COAL.
"""
response = rag_chain.invoke({"query": query})

# response = rag_chain.run(query)
# print("🔎 Answer:", response)
# print(response["result"])  # The answer text
# print('--------------------------------------------------------')
# print(response["source_documents"]) # List of source documents
# print(f"Responses recieved: {len(response["source_documents"])}")
#---------------------------------------------------------------------------------------------



print("Starting Reranking")

best_docs = cross_encoder_rerank(query=query, documents=response["source_documents"], top_k = 10)
reranked_retriever = StaticRetriever(docs=best_docs, k=10)

best_rag_chain = RetrievalQA.from_chain_type(llm=llm, 
                                            retriever=reranked_retriever,
                                            chain_type="stuff",  
                                            return_source_documents=True) 
best_response = best_rag_chain.invoke({"query": query})
print(best_response["result"]) 
print(best_response["source_documents"]) 
print(len(best_response["source_documents"]))