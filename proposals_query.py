from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from proposal import proposal
from helper_functions import llm, StaticRetriever, cross_encoder_rerank

embedding_model = HuggingFaceEmbeddings(model_name = "models/local_all-MiniLM-L6-v2")

vectorstore = Chroma(   
    persist_directory="./chromaDB",
    embedding_function=embedding_model,
    collection_name="proposal_collection"  # same name as before
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

print("Starting to Retreive to 10 Guidelines")


rag_chain = RetrievalQA.from_chain_type(llm=llm, 
                                        retriever=retriever,
                                        chain_type="stuff",  
                                        return_source_documents=True)  
query = f"""
    my proposal is {proposal}
    Find other proposals which might be similiar to my proposals
"""
response = rag_chain.invoke({"query": query})


print("Starting Reranking")

best_docs = cross_encoder_rerank(query=query, documents=response["source_documents"], top_k = 3)
reranked_retriever = StaticRetriever(docs=best_docs, k=3)

best_rag_chain = RetrievalQA.from_chain_type(llm=llm, 
                                            retriever=reranked_retriever,
                                            chain_type="stuff",  
                                            return_source_documents=True) 
best_response = best_rag_chain.invoke({"query": query})
print(best_response["result"])  # The answer text
print(best_response["source_documents"]) # List of source documents
print(len(best_response["source_documents"]))