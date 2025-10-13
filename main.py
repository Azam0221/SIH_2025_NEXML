import pandas as pd
import os
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_groq import ChatGroq
from langchain_community.graphs import Neo4jGraph
from dotenv import load_dotenv

load_dotenv()


groq_api_key = os.getenv("GROQ_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in .env file or environment variables.")

df = pd.read_csv('naccer_proposals_100_cleaned.csv')

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

llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")

llm_transformer = LLMGraphTransformer(llm=llm)

print("LLM and LLMGraphTransformer have been initialized.")

test_documents = documents[:5]

graph_documents = llm_transformer.convert_to_graph_documents(test_documents)
first_graph_doc = graph_documents[0]

print("\n--- Example Extracted Graph ---")
print("Nodes:")
for node in first_graph_doc.nodes:
    print(f"  - {node}")

print("\nRelationships:")
for rel in first_graph_doc.relationships:
    print(f"  - {rel}")
print("-----------------------------\n")
