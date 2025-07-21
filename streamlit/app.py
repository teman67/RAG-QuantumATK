# app.py
import streamlit as st
from rag_qa import get_rag_chain_with_key
from langchain_huggingface import HuggingFaceEmbeddings

st.set_page_config(page_title="QuantumATK Assistant", layout="wide")

st.title("🔬 QuantumATK Documentation Assistant")
st.markdown("Ask a question to get relevant info from the [QuantumATK docs](https://docs.quantumatk.com)")

# 🔑 API Key input
api_key = st.text_input("🔐 Enter your OpenAI API Key", type="password")

if api_key:
    query = st.text_input("🧠 Enter your question")

    if query:
        with st.spinner("Retrieving answer..."):
            try:
                chain = get_rag_chain_with_key(api_key)
                result = chain.run(query)
                st.markdown("### 💬 Answer")
                st.write(result)
            except Exception as e:
                st.error(f"Error: {e}")
else:
    st.info("Please enter your OpenAI API key to continue.")
