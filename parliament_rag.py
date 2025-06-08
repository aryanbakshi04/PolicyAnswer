import streamlit as st
import requests
from bs4 import BeautifulSoup
from io import BytesIO
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import VertexAI
from agno.agent import Agent
from agno.models.google import Gemini

# Constants
BASE_URL = "https://legalaffairs.gov.in/parliament-qa"
PDF_CACHE_DIR = "pdf_cache"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL_NAME = "gemini-2.0-flash-exp"  # adjust as per available model

api_key = os.getenv("GOOGLE_API_KEY")

os.environ["GOOGLE_API_KEY"] = api_key

# Ensure cache directory exists
os.makedirs(PDF_CACHE_DIR, exist_ok=True)

@st.cache_data
def fetch_pdf_links(max_pages=5):
    pdf_links = []
    for page in range(max_pages):
        params = {
            'field_house_tid': 'All',
            'field_question_type_tid': 'All',
            'field_question_no_value': '',
            'title': '',
            'page': page
        }
        resp = requests.get(BASE_URL, params=params)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        for a in soup.select('a.download-pdf'):
            href = a.get('href')
            if href and href.lower().endswith('.pdf'):
                pdf_links.append(href)
    return list(set(pdf_links))

@st.cache_data
def download_pdf(url):
    filename = os.path.join(PDF_CACHE_DIR, os.path.basename(url))
    if not os.path.exists(filename):
        r = requests.get(url)
        r.raise_for_status()
        with open(filename, 'wb') as f:
            f.write(r.content)
    return filename

@st.cache_resource
def build_vectorstore(pdf_urls):
    docs = []
    for url in pdf_urls:
        path = download_pdf(url)
        loader = PyPDFLoader(path)
        docs.extend(loader.load())
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    vectordb = FAISS.from_documents(chunks, embeddings)
    return vectordb

@st.cache_resource
def get_qa_chain(vectordb):
    # Ensure GOOGLE_APPLICATION_CREDENTIALS env var is set for Vertex AI
    llm = VertexAI(model_name=GEMINI_MODEL_NAME, temperature=0)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever()
    )
    return qa

# Streamlit UI
st.title("Parliamentary QA Assistant (Agentic AI with Gemini)")

st.sidebar.header("Configuration")
max_pages = st.sidebar.number_input("Max pages to crawl", min_value=1, max_value=20, value=5)

st.write("This tool answers parliamentary questions based on past ministerial responses, powered by Google Gemini via Vertex AI.")

if 'vectordb' not in st.session_state:
    with st.spinner("Fetching PDF links..."):
        pdf_links = fetch_pdf_links(max_pages)
    with st.spinner(f"Downloading and indexing {len(pdf_links)} PDFs..."):
        st.session_state.vectordb = build_vectorstore(pdf_links)
    st.session_state.qa = get_qa_chain(st.session_state.vectordb)

question = st.text_area("Enter the parliamentary question:")
minister = st.text_input("Name of the Minister (e.g. Shri Rajmohan Unnithan):")

if st.button("Get Answer"):
    if not question.strip() or not minister.strip():
        st.error("Please provide both a question and minister name.")
    else:
        prompt = (
            f"You are answering on behalf of {minister}. "
            "Provide a formal, solution-oriented answer focused on public interest and welfare. "
            f"Question: {question}"
        )
        with st.spinner("Generating answer with Gemini..."):
            answer = st.session_state.qa.run(prompt)
        st.subheader("Answer")
        st.write(answer)
