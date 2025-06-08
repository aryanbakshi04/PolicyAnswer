from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import requests
from bs4 import BeautifulSoup
import os
from io import BytesIO
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from agno.agent import Agent
from agno.models.google import Gemini

# Constants
BASE_URL = "https://legalaffairs.gov.in/parliament-qa"
PDF_CACHE_DIR = "pdf_cache"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL_NAME = "gemini-2.0-flash-exp"

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
        loader = UnstructuredPDFLoader(path)
        docs.extend(loader.load())
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    vectordb = FAISS.from_documents(chunks, embeddings)
    return vectordb

@st.cache_resource
def init_agent():
    # Instantiate a Gemini-based agent
    agent = Agent(
        model=Gemini(id=GEMINI_MODEL_NAME),
        description="Answers parliamentary questions based on retrieved ministerial responses.",
        instructions=[
            "Use the provided context extracted from past minister PDFs.",
            "Answer formally on behalf of the specified minister, focusing on public welfare and solutions."
        ],
        show_tool_calls=False,
        markdown=False
    )
    return agent

# Streamlit UI
st.title("Parliamentary QA Assistant (Agentic AI with Gemini)")
st.sidebar.header("Configuration")
max_pages = st.sidebar.number_input("Max pages to crawl", min_value=1, max_value=20, value=5)

st.write("This tool answers parliamentary questions in the style of ministers, using Google Gemini.")

if 'vectordb' not in st.session_state:
    with st.spinner("Indexing archival PDFs..."):
        links = fetch_pdf_links(max_pages)
        st.session_state.vectordb = build_vectorstore(links)
        st.session_state.agent = init_agent()

question = st.text_area("Enter the parliamentary question:")
minister = st.text_input("Minister Name (e.g. Shri Rajmohan Unnithun):")

if st.button("Get Answer"):
    if not question.strip() or not minister.strip():
        st.error("Provide both the question and minister name.")
    else:
        # Retrieve top 4 relevant chunks
        docs = st.session_state.vectordb.similarity_search(question, k=4)
        context = "\n\n".join([d.page_content for d in docs])
        prompt = (
            f"Context:\n{context}\n\n"
            f"Answer as {minister}: Provide a formal, solution-oriented response focused on public interest. "
            f"Question: {question}"
        )
        with st.spinner("Generating response with Gemini..."):
            result = st.session_state.agent.run(prompt)
        st.subheader("Answer")
        st.write(result)
