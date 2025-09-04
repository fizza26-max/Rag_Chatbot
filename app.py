
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
import tempfile
import os

# Page setup
st.set_page_config(page_title="AI Document Search", page_icon="📄")

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #ffffff;
        color: #0d47a1;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stButton>button {
        background-color: #0d47a1;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #1565c0;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📄 AI Document Search (RAG Chatbot)")
st.write("Upload a PDF document and ask questions about its content.")

# Globals (session state)
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# Upload PDF
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file is not None:
    if st.button("Process PDF"):
        with st.spinner("Processing PDF..."):
            # Save temporarily
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            # Load and split text
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = text_splitter.split_documents(documents)

            # Embeddings
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            st.session_state.vectorstore = FAISS.from_documents(docs, embeddings)
            retriever = st.session_state.vectorstore.as_retriever()

            # LLM (local transformer, no API key needed)
            generator = pipeline("text2text-generation", model="google/flan-t5-base")
            llm = HuggingFacePipeline(pipeline=generator)

            # QA chain
            st.session_state.qa_chain = RetrievalQA.from_chain_type(
                llm=llm, chain_type="stuff", retriever=retriever
            )

            os.remove(tmp_path)
            st.success(f"✅ PDF '{uploaded_file.name}' processed and indexed!")

# Ask questions
query = st.text_input("Ask a question about the document:")

if query:
    if st.button("Get Answer"):
        if st.session_state.qa_chain is None:
            st.error("Please upload and process a PDF first.")
        else:
            with st.spinner("Generating answer..."):
                answer = st.session_state.qa_chain.run(query)
                st.markdown(f"### 🤖 Answer:\n{answer}")
