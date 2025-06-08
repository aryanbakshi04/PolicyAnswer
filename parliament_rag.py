try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import os
import streamlit as st
import requests
from bs4 import BeautifulSoup

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS

from agno.agent import Agent
from agno.models.google import Gemini

BASE_URL = "https://legalaffairs.gov.in/parliament-qa"
PDF_CACHE_DIR = "pdf_cache"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL_NAME = "gemini-2.0-flash-exp"

os.makedirs(PDF_CACHE_DIR, exist_ok=True)

@st.cache_data
def fetch_pdf_links(max_pages=5):
    pdf_links = set()
    import urllib.parse
    for page in range(max_pages):
        url = f"{BASE_URL}?field_house_tid=All&field_question_type_tid=All&field_question_no_value=&title=&page={page}"
        resp = requests.get(url)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        for a in soup.find_all('a', href=True):
            href = a['href']
            if href.lower().endswith('.pdf'):
                full_url = urllib.parse.urljoin(BASE_URL, href)
                pdf_links.add(full_url)
    return list(pdf_links)

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
    if not pdf_urls:
        st.error("No PDF links found. Please check the source URL or page settings.")
        return None
    docs = []
    for url in pdf_urls:
        path = download_pdf(url)
        loader = PyPDFLoader(path)
        docs_loaded = loader.load()
        for doc in docs_loaded:
            doc.metadata['source_url'] = url
        docs.extend(docs_loaded)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    if not chunks:
        st.error("No text extracted from PDFs. Cannot build index.")
        return None
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    vectordb = FAISS.from_documents(chunks, embeddings)
    return vectordb

@st.cache_resource
def init_agent():
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

st.title("Parliamentary QA Assistant (Gemini RAG)")
st.sidebar.header("Settings")
max_pages = st.sidebar.number_input("Number of pages to crawl for PDFs", 1, 20, 5)

if 'vectordb' not in st.session_state:
    with st.spinner("Indexing past minister PDFs..."):
        pdf_links = fetch_pdf_links(max_pages)
        vectordb = build_vectorstore(pdf_links)
        if vectordb is None:
            st.stop()
        st.session_state.vectordb = vectordb
        st.session_state.agent = init_agent()

question = st.text_area("Parliamentary Question:")
minister = st.text_input("Answering Minister (e.g., Shri Rajmohan Unnithun):")

if st.button("Generate Answer"):
    if not question.strip() or not minister.strip():
        st.error("Both a question and minister name are required.")
    else:
        docs = st.session_state.vectordb.similarity_search(question, k=4)
        if not docs:
            st.error("No relevant context found for the question.")
        else:
            context = "\n\n".join([doc.page_content for doc in docs])
            prompt = (
                f"Context:\n{context}\n\n"
                f"Answer as {minister}: Provide a formal, solution-oriented response focused on public interest. Question: {question}"
            )
            with st.spinner("Generating response..."):
                response = st.session_state.agent.run(prompt)

            # Extract clean content and PDF citations
            answer = response.content if hasattr(response, 'content') else str(response)
            source_urls = list({doc.metadata.get('source_url') for doc in docs if 'source_url' in doc.metadata})

            st.subheader("📝 Answer")
            st.write(answer)

            if source_urls:
                st.subheader("📄 Source PDF(s)")
                for link in source_urls:
                    st.markdown(f"- [View PDF]({link})")
            else:
                st.info("No source PDFs found.")
