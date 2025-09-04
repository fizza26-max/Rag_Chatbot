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
import docx
import easyocr

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
st.write("Upload a document (PDF, TXT, DOCX, or Image) and ask questions about its content.")

# Globals (session state)
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

def load_txt(file):
    return [{"page_content": file.read().decode("utf-8"), "metadata": {}}]

def load_docx(file):
    doc = docx.Document(file)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return [{"page_content": "\n".join(full_text), "metadata": {}}]

def load_image(file):
    # Use easyocr to extract text from image bytes
    reader = easyocr.Reader(['en'], gpu=False)  # disable GPU for compatibility
    # Save temporarily to disk because easyocr reads from path or numpy array
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
        tmp_img.write(file.read())
        tmp_path = tmp_img.name
    result = reader.readtext(tmp_path, detail=0, paragraph=True)
    os.remove(tmp_path)
    text = "\n".join(result)
    return [{"page_content": text, "metadata": {}}]

uploaded_file = st.file_uploader("Upload Document", type=["pdf", "txt", "docx", "png", "jpg", "jpeg"])

if uploaded_file is not None:
    if st.button("Process Document"):
        with st.spinner("Processing document..."):
            suffix = os.path.splitext(uploaded_file.name)[1].lower()

            # Load documents based on file type
            if suffix == ".pdf":
                # Save temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                loader = PyPDFLoader(tmp_path)
                documents = loader.load()
                os.remove(tmp_path)
            elif suffix == ".txt":
                documents = load_txt(uploaded_file)
            elif suffix == ".docx":
                documents = load_docx(uploaded_file)
            elif suffix in [".png", ".jpg", ".jpeg"]:
                documents = load_image(uploaded_file)
            else:
                st.error("Unsupported file type!")
                st.stop()

            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = text_splitter.split_documents(documents)

            # Embeddings - more accurate model
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
            st.session_state.vectorstore = FAISS.from_documents(docs, embeddings)
            retriever = st.session_state.vectorstore.as_retriever()

            # LLM - larger and more accurate open source model
            generator = pipeline(
                "text2text-generation",
                model="google/flan-t5-large",
                device=0 if os.environ.get("CUDA_VISIBLE_DEVICES") else -1,
                max_length=512,
                do_sample=False,
            )
            llm = HuggingFacePipeline(pipeline=generator)

            # QA chain
            st.session_state.qa_chain = RetrievalQA.from_chain_type(
                llm=llm, chain_type="stuff", retriever=retriever
            )

            st.success(f"✅ Document '{uploaded_file.name}' processed and indexed!")

query = st.text_input("Ask a question about the document:")

if query:
    if st.button("Get Answer"):
        if st.session_state.qa_chain is None:
            st.error("Please upload and process a document first.")
        else:
            with st.spinner("Generating answer..."):
                answer = st.session_state.qa_chain.run(query)
                st.markdown(f"### 🤖 Answer:\n{answer}")
