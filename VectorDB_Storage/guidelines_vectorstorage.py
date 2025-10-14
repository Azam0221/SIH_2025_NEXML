from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def read_doc(directory):
    file_loader=PyPDFDirectoryLoader(directory)
    documents=file_loader.load()
    return documents

def chunk_data(docs, chunk_size = 800, chunk_overlap = 50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap = chunk_overlap)
    doc = text_splitter.split_documents(docs)
    return doc



doc=read_doc('./documents/')
chunked_documents = chunk_data(doc)

embedding_model = HuggingFaceEmbeddings(model_name = "models/local_all-MiniLM-L6-v2")
persist_directory = "./GuidelinesDB"
vectorstore = Chroma.from_documents(chunked_documents,
                                    embedding_model,
                                    collection_name="guidelines_collection",
                                    persist_directory=persist_directory)

