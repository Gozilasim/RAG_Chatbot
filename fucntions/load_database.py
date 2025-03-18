import pickle
from langchain.vectorstores import VectorStore
from langchain.embeddings import HuggingFaceEmbeddings



def load_vector_store():
    # load vectorstore.pkl file from local
    vectorstore_file = "RAG\multi-qa-MiniLM-L6-cos-v1.pkl"

    with open(vectorstore_file, "rb") as f:
        global vectorstore
        db: VectorStore = pickle.load(f)


    return db

def load_vector_databse():

    # load vectorstore.pkl file from local
    vectorstore_file = "RAG\multi-qa-MiniLM-L6-cos-v1.pkl"

    with open(vectorstore_file, "rb") as f:
        vectorStore = pickle.load(f)
    
    return vectorStore