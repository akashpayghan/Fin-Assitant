
import streamlit as st
from openai import OpenAI
import os, chromadb, uuid
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

st.set_page_config(page_title="Personal Finance Assistant", layout="wide")
st.title("Personal Finance Assistant")

init_error = None
rag_engine = None
doc_processor = None
try:
    from document_processor import DocumentProcessor
    from rag_engine import RAGEngine
    doc_processor = DocumentProcessor()
    rag_engine = RAGEngine()  # now protected
except Exception as e:
    init_error = str(e)

if init_error:
    st.error(f"Startup error: {init_error}")
    st.stop()



class RAGEngine:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.chat_model = os.getenv("CHAT_MODEL", "gpt-4o-mini")
        self.chroma = chromadb.PersistentClient(path="./chroma_db")
        try:
            self.col = self.chroma.get_collection("financial_documents")
        except:
            self.col = self.chroma.create_collection("financial_documents")

    def generate_embeddings(self, texts):
        resp = self.client.embeddings.create(model=self.embedding_model, input=texts)
        return [d.embedding for d in resp.data]

    def chat(self, messages):
        return self.client.chat.completions.create(
            model=self.chat_model, messages=messages, temperature=0.3, max_tokens=500
        ).choices[0].message.content



uploaded = st.sidebar.file_uploader("Upload PDFs/CSVs/TXT", type=["pdf","csv","txt"], accept_multiple_files=True)
if st.sidebar.button("Process") and uploaded:
    with st.spinner("Indexing..."):
        chunks = []
        for f in uploaded:
            chunks.extend(doc_processor.process_document(f))
        if chunks:
            rag_engine.add_documents(chunks)
st.header("Ask a question")
q = st.chat_input("Type here…")
if q:
    ctx = rag_engine.search_similar_documents(q, n_results=3)
    st.write(rag_engine.generate_response(q, ctx))
