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
from docx import Document
import pytesseract
from PIL import Image

# Page setup
st.set_page_config(page_title="AI Document Search", page_icon="📄")

# Custom CSS
st.markdown(
    """
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
    """,
    unsafe_allow_html=True,
)

st.title("📄 AI Document Search (RAG Chatbot)")
st.write("Upload a document (PDF, TXT, DOCX, or Image) and ask questions about its content.")

# Globals (session state)
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

def extract_text_from_docx(path):
    doc = Document(path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return "\n".join(full_text)

def extract_text_from_image(image_path):
    try:
        img = Image.open(image_path)
        # Use pytesseract to do OCR on the image
        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        st.error(f"Failed to process image OCR: {e}")
        return ""

def process_uploaded_file(uploaded_file_path, file_extension):
    text = ""
    try:
        if file_extension == ".pdf":
            loader = PyPDFLoader(uploaded_file_path)
            documents = loader.load()
            # Extract combined text
            text = "\n".join([doc.page_content for doc in documents])
        elif file_extension == ".txt":
            with open(uploaded_file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        elif file_extension == ".docx":
            text = extract_text_from_docx(uploaded_file_path)
        elif file_extension in [".jpg", ".jpeg", ".png"]:
            text = extract_text_from_image(uploaded_file_path)
        else:
            st.error(f"Unsupported file type: {file_extension}")
            return None
    except Exception as e:
        st.error(f"Failed to extract text from file: {e}")
        return None
    return text

uploaded_file = st.file_uploader(
    "Upload file", type=["pdf", "txt", "docx", "jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    if st.button("Process File"):
        with st.spinner("Processing file..."):
            try:
                suffix = os.path.splitext(uploaded_file.name)[1].lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                # Extract text from file
                raw_text = process_uploaded_file(tmp_path, suffix)
                if not raw_text or raw_text.strip() == "":
                    st.error("No text could be extracted from the file.")
                    os.remove(tmp_path)
                    st.session_state.qa_chain = None
                    st.session_state.vectorstore = None
                else:
                    # Chunk text
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000, chunk_overlap=200
                    )
                    fake_docs = [{"page_content": raw_text}]
                    docs = text_splitter.split_documents(fake_docs)

                    # Embeddings - more accurate model
                    embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L12-v2"
                    )
                    st.session_state.vectorstore = FAISS.from_documents(docs, embeddings)
                    retriever = st.session_state.vectorstore.as_retriever()

                    # LLM - stronger open source model from HF Hub
                    generator = pipeline(
                        "text2text-generation",
                        model="google/flan-t5-xl",
                        device=0 if torch.cuda.is_available() else -1,
                        max_length=512,
                        do_sample=False,
                    )
                    llm = HuggingFacePipeline(pipeline=generator)

                    # QA chain
                    st.session_state.qa_chain = RetrievalQA.from_chain_type(
                        llm=llm, chain_type="stuff", retriever=retriever
                    )

                    st.success(f"✅ File '{uploaded_file.name}' processed and indexed!")
                os.remove(tmp_path)
            except Exception as e:
                st.error(f"Failed to process file: {e}")
                st.session_state.qa_chain = None
                st.session_state.vectorstore = None

# Ask questions
query = st.text_input("Ask a question about the document:")

if query:
    if st.button("Get Answer"):
        if st.session_state.qa_chain is None:
            st.error("Please upload and process a document first.")
        else:
            with st.spinner("Generating answer..."):
                answer = st.session_state.qa_chain.run(query)
                st.markdown(f"### 🤖 Answer:\n{answer}")
