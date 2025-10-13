# SIH 2025 NEXML

## Project Deliverables

1. Centralized Project Repository: The establishment and maintenance of a structured digital database containing all past and ongoing R&D projects to serve as a centralized knowledge base.
2. Automated Novelty Assessment: A system for benchmarking submitted proposals against the centralized repository to quantitatively assess the originality and novelty of the proposed research.
3. Automated Financial Compliance Verification: A module to perform automated financial checks, ensuring that the budgetary allocations within each proposal are in strict alignment with the mandated S&T funding guidelines.
4. Objective Evaluation Score Generation: An intelligent system to generate a comprehensive and objective evaluation score for each proposal, derived from the novelty assessment, financial compliance check, and other relevant parameters.



## General Flow for RAG

All PDF files can  be centrally stored at a designated location.  
> Note: The software team decide what to do.

---

### Data Ingestion Pipeline

The ingestion pipeline processes and prepares PDF data for retrieval by performing the following steps:

1. **PDF Loading**
   - Use `PDFLoader` from **Langchain** to extract raw document structure (metadata + content) from each PDF file.

2. **Text Chunking**
   - Apply `TextSplitter` to divide the extracted content into manageable and semantically meaningful chunks.

3. **Embeddings Generation**
   - Use **Sentence Transformers** to convert each chunk of text into high-dimensional vector embeddings.

4. **Vector Store Integration**
   - Store the generated embeddings using a vector database like **FAISS** or **Chroma DB**.
---

### Retrieval Pipeline

This pipeline handles user queries and fetches relevant information from the ingested documents:

1. **Query Embedding**
   - Convert the user's query into an embedding using the same **Sentence Transformer** model.

2. **Similarity Search**
   - Query the vector database to find top-matching document chunks based on cosine similarity.

3. **Context Filtering**
   - For each retrieved document chunk:
     - If **similarity_score > threshold**, add it to the context window.

4. **LLM Integration**
   - Use **Groq LLM via Langchain** for natural language generation and summarization.

5. **Prompt Construction**
   - Combine:
     - Retrieved context chunks
     - Formatted system/user prompt
   - Pass this combined prompt to the LLM.

6. **Response Generation**
   - LLM generates a relevant and coherent answer based on the provided context.


## Approach

1. Centralized Project Repository: 
    - Afifa,Azam decide best way to implement this. 
    - Should the centralized repo be a separate db or should we keep the vector db as central repo?


2. Automated Novelty Assessment: 
    - Basically we will try to find if there is any document in the vector db which is similar to the current proposal
    - If there is any such proposal then we  will send that as context to the llm
    - In the prompt we will ask it to assess the novelty of the current proposal based on the given context

3. Automated Financial Compliance Verification:
    - I think here we can simply send the content of the mandated S&T funding guidelines along with the contetn of the proposal to the llm and ask it to generate a financial compliance report or smth.

4. Objective Evaluation Score Generation:
    - For this I am think we can send content of the repo,the novelty assessment and also the financial reposr. Also we should compare the current proposal somehow with the rest of the proposals and generate relative evaluation scores. This will be better than arbitrary scores I think. Though we will have to decide how to actually go about this one.



### Setup

1. **Create and activate virtual environment using `uv`:**

   ```bash
   uv venv
   source .venv/bin/activate
   uv pip install -r requirements.txt
   ```

2. **Create a .env file in the project root with the following:**
   ```
      GROQ_API_KEY=your_groq_api_key
      NEO4J_URI=your_neo4j_uri   
      NEO4J_USERNAME=your_username
      NEO4J_PASSWORD=your_password
   ```
