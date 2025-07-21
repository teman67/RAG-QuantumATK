# rag_qa.py


from langchain_huggingface import HuggingFaceEmbeddings  # NEW preferred import
from langchain_community.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

def get_rag_chain_with_key(openai_api_key: str):
    # Initialize embedding model (on CPU)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    # Load FAISS vector store from local disk
    vectorstore = FAISS.load_local(
        "quantumatk_faiss",
        embeddings,
        allow_dangerous_deserialization=True
    )

    # Setup retriever
    retriever = vectorstore.as_retriever(search_type="similarity", k=5)

    # Load LLM (OpenAI)
    llm = ChatOpenAI(
        model="gpt-4o-mini",  # or "gpt-4o-mini" if available and supported
        temperature=0,
        openai_api_key=openai_api_key
    )

    # Build QA chain using the retriever and LLM
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True  # Optional: to show sources in UI
    )

    return qa_chain
