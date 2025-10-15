from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from helper_functions import llm, cross_encoder_rerank, StaticRetriever
from tqdm import tqdm
import json

with open(r'data\ragas_eval.json', 'r') as file:
    data = json.load(file)

embedding_model = HuggingFaceEmbeddings(model_name = "models/local_all-MiniLM-L6-v2")
vectorstore = Chroma(   
    persist_directory="./chromaDB",
    embedding_function=embedding_model,
    collection_name="proposal_collection"  # same name as before
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
rag_chain = RetrievalQA.from_chain_type(llm=llm, 
                                        retriever=retriever,
                                        chain_type="stuff",  
                                        return_source_documents=True)  

for item in tqdm(data):
    query = f"{item["question"]}"
    response = rag_chain.invoke({"query": query})
    best_docs = cross_encoder_rerank(query=query, documents=response["source_documents"], top_k = 2)
    item["contexts"]=[doc.page_content for doc in best_docs]
    reranked_retriever = StaticRetriever(docs=best_docs, k=2)
    best_rag_chain = RetrievalQA.from_chain_type(llm=llm, 
                                                retriever=reranked_retriever,
                                                chain_type="stuff",  
                                                return_source_documents=False) 
    best_response = best_rag_chain.invoke({"query": query})
    
    for key, val in best_response.items():
        if key=='result':
            item['answer'] = val
    

with open('data/ragas_eval_filled.json', 'w') as f:
    json.dump(data, f, indent=4)
        

print("JSON Updated")











































# rag_chain = RetrievalQA.from_chain_type(llm=llm, 
#                                         retriever=retriever,
#                                         chain_type="stuff",  
#                                         return_source_documents=True)  
# query = f"""
#     my proposal is {proposal}
#     Find other proposals which might be similiar to my proposals
# """
# response = rag_chain.invoke({"query": query})

# print(response["result"])
# print(f"Responses recieved: {len(response["source_documents"])}")

# print("Starting Reranking")

# # best_docs = return_best_docs(docs=response["source_documents"], query=query)
# best_docs = cross_encoder_rerank(query=query, documents=response["source_documents"], top_k = 3)
# reranked_retriever = StaticRetriever(docs=best_docs, k=3)

# best_rag_chain = RetrievalQA.from_chain_type(llm=llm, 
#                                             retriever=reranked_retriever,
#                                             chain_type="stuff",  
#                                             return_source_documents=True) 
# best_response = best_rag_chain.invoke({"query": query})
# print(best_response["result"])  # The answer text
# # print('--------------------------------------------------------')
# # print(best_response["source_documents"]) # List of source documents
# print(len(best_response["source_documents"]))