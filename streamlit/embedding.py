# embedding.py (updated for LangChain >= 0.2)
from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def load_and_embed(docs_dir="quantumatk_docs"):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = []
    for fname in os.listdir(docs_dir):
        loader = TextLoader(os.path.join(docs_dir, fname), encoding="utf-8")
        docs.extend(splitter.split_documents(loader.load()))

    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("quantumatk_faiss")
    

if __name__ == "__main__":
    load_and_embed()
