import streamlit as st
import os
import requests
import numpy as np
import faiss

from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI SaaS Assistant", layout="wide")

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or st.secrets.get("OPENROUTER_API_KEY")
FIREBASE_API_KEY = os.getenv("FIREBASE_API_KEY") or st.secrets.get("FIREBASE_API_KEY")

# ---------------- LOGIN ----------------
def login(email, password):
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_API_KEY}"
    
    payload = {
        "email": email,
        "password": password,
        "returnSecureToken": True
    }

    res = requests.post(url, json=payload)
    return res.json()

if "user" not in st.session_state:
    st.session_state.user = None

if not st.session_state.user:

    st.title("🔐 Login to AI Assistant")

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        result = login(email, password)

        if "idToken" in result:
            st.session_state.user = email
            st.success("Logged in!")
            st.rerun()
        else:
            st.error("Invalid credentials")

    st.stop()

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("📂 Upload PDFs")

    uploaded_files = st.file_uploader(
        "Upload PDFs", type="pdf", accept_multiple_files=True
    )

    if st.button("🔄 Reset"):
        st.session_state.clear()
        st.rerun()

# ---------------- MAIN ----------------
st.title(f"🤖 Welcome {st.session_state.user}")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------- PROCESS PDFs ----------------
if uploaded_files and "index" not in st.session_state:

    with st.spinner("Processing documents..."):

        all_chunks = []
        chunk_sources = []

        for file in uploaded_files:
            reader = PdfReader(file)

            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()

                if text:
                    chunk_size = 300
                    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

                    for chunk in chunks:
                        all_chunks.append(chunk)
                        chunk_sources.append({
                            "file": file.name,
                            "page": page_num + 1,
                            "text": chunk
                        })

        embeddings = model.encode(all_chunks)

        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings))

        st.session_state.index = index
        st.session_state.chunks = all_chunks
        st.session_state.sources = chunk_sources

        st.success("✅ Documents ready!")

# ---------------- SHOW CHAT ----------------
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- INPUT ----------------
question = st.chat_input("Ask something about your documents...")

# ---------------- LIMIT (FREE PLAN) ----------------
if "usage" not in st.session_state:
    st.session_state.usage = 0

LIMIT = 5

# ---------------- QA ----------------
if question and "index" in st.session_state:

    if st.session_state.usage >= LIMIT:
        st.warning("⚠️ Free limit reached. Upgrade coming soon!")
        st.stop()

    st.session_state.usage += 1

    st.session_state.chat_history.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    index = st.session_state.index
    chunks = st.session_state.chunks
    sources = st.session_state.sources

    with st.spinner("Thinking..."):

        q_embedding = model.encode([question])
        _, indices = index.search(np.array(q_embedding), 3)

        top_chunks = [chunks[i] for i in indices[0]]
        top_sources = [sources[i] for i in indices[0]]

        context = "\n".join(top_chunks)

        data = {
            "model": "openai/gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "Answer using only the provided context."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion:{question}"}
            ]
        }

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json=data
        )

        result = response.json()

        if "choices" in result:
            answer = result["choices"][0]["message"]["content"]

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer
            })

            with st.chat_message("assistant"):
                st.markdown(answer)

                st.markdown("### 📚 Sources")
                for src in top_sources:
                    st.markdown(f"- 📄 {src['file']} (page {src['page']})")
                    with st.expander("View text"):
                        st.write(src["text"])

                st.download_button(
                    "⬇️ Download Answer",
                    data=answer,
                    file_name="answer.txt"
                )

        else:
            st.error(result)