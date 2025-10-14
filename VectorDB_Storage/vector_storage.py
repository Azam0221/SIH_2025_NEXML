import pandas as pd
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


df = pd.read_csv(r'data\naccer_proposals_100_cleaned.csv')

documents = []
for index, row in df.iterrows():
    page_content = (
        f"Proposal Title: {row['Title']}\n"
        f"Principal Investigator: {row['PI_Name']}\n"
        f"Institution: {row['Institution']}\n"
        f"Research Area: {row['Research_Area']}\n"
        f"Keywords: {row['Keywords']}\n"
        f"Abstract: {row['Abstract']}"
    )
    
    doc = Document(page_content=page_content, metadata={"proposal_id": row['Proposal_ID']})
    documents.append(doc)


embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
persist_directory = "./chromaDB"
vectorstore = Chroma.from_documents(documents,
                                    embedding_model,
                                    collection_name="proposal_collection",
                                    persist_directory=persist_directory)
