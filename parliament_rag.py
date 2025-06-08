# Optional dotenv support: load environment variables if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available; rely on environment variables only
    pass

import os
import streamlit as st  # Streamlit for UI
import requests  # HTTP requests to fetch web pages and PDFs
from bs4 import BeautifulSoup  # HTML parsing for PDF link extraction
from io import BytesIO  # In-memory byte streams for file I/O

# LangChain for document processing and embeddings
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS

# Agent and Gemini model for generation
from agno.agent import Agent
from agno.models.google import Gemini

# --- Configuration Constants ---
BASE_URL = "https://legalaffairs.gov.in/parliament-qa"  # Parliamentary QA portal
PDF_CACHE_DIR = "pdf_cache"  # Local directory to cache downloaded PDFs
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Embedding model
GEMINI_MODEL_NAME = "gemini-2.0-flash-exp"  # Google Gemini model ID

# Ensure the PDF cache directory exists
ios.makedirs(PDF_CACHE_DIR, exist_ok=True)

# --- Helper Functions ---
@st.cache_data
def fetch_pdf_links(max_pages=5):
    """
    Crawl the QA portal pages and collect unique PDF URLs.
    """
    pdf_links = []  # List to store all discovered PDF links
    for page in range(max_pages):
        params = {
            'field_house_tid': 'All',
            'field_question_type_tid': 'All',
            'field_question_no_value': '',
            'title': '',
            'page': page
        }
        # Fetch the HTML of the page
        resp = requests.get(BASE_URL, params=params)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        # Extract all <a> tags pointing to PDFs
        for a in soup.select('a.download-pdf'):
            href = a.get('href')
            if href and href.lower().endswith('.pdf'):
                pdf_links.append(href)
    # Return unique links
    return list(set(pdf_links))

@st.cache_data
def download_pdf(url):
    """
    Download a PDF from a URL, caching it locally to avoid re-downloading.
    """
    filename = os.path.join(PDF_CACHE_DIR, os.path.basename(url))
    if not os.path.exists(filename):
        r = requests.get(url)
        r.raise_for_status()
        with open(filename, 'wb') as f:
            f.write(r.content)
    return filename

@st.cache_resource
def build_vectorstore(pdf_urls):
    """
    Load PDFs, split into chunks, and index with FAISS embeddings for retrieval.
    """
    docs = []
    for url in pdf_urls:
        path = download_pdf(url)
        # Use UnstructuredPDFLoader to read PDF contents into LangChain docs
        loader = UnstructuredPDFLoader(path)
        docs.extend(loader.load())
    # Split into manageable chunks for embedding
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    # Initialize embeddings and FAISS vector store
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    vectordb = FAISS.from_documents(chunks, embeddings)
    return vectordb

@st.cache_resource
def init_agent():
    """
    Initialize the Agent with the specified Gemini model for generation.
    """
    return Agent(
        model=Gemini(id=GEMINI_MODEL_NAME),
        description="Answers parliamentary questions based on retrieved context.",
        instructions=[
            "Use the retrieved context from past ministerial answers.",
            "Respond formally on behalf of the specified minister with focus on public welfare."
        ],
        show_tool_calls=False,
        markdown=False
    )

# --- Streamlit App UI ---
st.title("Parliamentary QA Assistant (Gemini RAG)")
st.sidebar.header("Settings")
max_pages = st.sidebar.number_input(
    "Number of pages to crawl for PDFs", min_value=1, max_value=20, value=5
)

st.write(
    "Enter a parliamentary question and minister’s name, and get a formally phrased answer based on past answers."
)

# Build vector store and agent on first run
if 'vectordb' not in st.session_state:
    with st.spinner("Indexing past minister PDFs..."):
        pdf_links = fetch_pdf_links(max_pages)
        st.session_state.vectordb = build_vectorstore(pdf_links)
        st.session_state.agent = init_agent()

# Inputs
question = st.text_area("Parliamentary Question:")
minister = st.text_input("Answering Minister (e.g., Shri Rajmohan Unnithun):")

# Trigger generation
if st.button("Generate Answer"):
    if not question.strip() or not minister.strip():
        st.error("Both a question and minister name are required.")
    else:
        # Retrieve top-k relevant chunks for RAG
        docs = st.session_state.vectordb.similarity_search(question, k=4)
        # Concatenate retrieved text as context
        context = "\n\n".join([doc.page_content for doc in docs])
        # Build the final prompt including context and instructions
        prompt = (
            f"Context:\n{context}\n\n"
            f"Answer as {minister}: Provide a formal, solution-oriented response focused on public interest. "
            f"Question: {question}"
        )
        # Generate with Gemini Agent
        with st.spinner("Generating response..."):
            response = st.session_state.agent.run(prompt)
        # Display result
        st.subheader("Generated Answer")
        st.write(response)
