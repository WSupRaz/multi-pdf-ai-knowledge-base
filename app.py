import streamlit as st
import os
import requests
import numpy as np
import faiss

from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load API key
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

url = "https://openrouter.ai/api/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

st.title("📄 AI Document Assistant")

uploaded_files = st.file_uploader(
    "Upload PDF files", type="pdf", accept_multiple_files=True
)

question = st.text_input("Ask a question about the documents")

if uploaded_files and question:

    model = SentenceTransformer("all-MiniLM-L6-v2")

    all_chunks = []

    for file in uploaded_files:

        reader = PdfReader(file)

        text = ""

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text

        chunk_size = 300
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

        all_chunks.extend(chunks)

    embeddings = model.encode(all_chunks)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)

    index.add(np.array(embeddings))

    q_embedding = model.encode([question])

    k = 3
    distances, indices = index.search(np.array(q_embedding), k)

    top_chunks = [all_chunks[i] for i in indices[0]]

    context = "\n".join(top_chunks)

    data = {
        "model": "openai/gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "Answer using only the provided context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:{question}"}
        ]
    }

    response = requests.post(url, headers=headers, json=data)

    result = response.json()

    if "choices" in result:
        answer = result["choices"][0]["message"]["content"]
        st.write("### AI Answer")
        st.write(answer)
    else:
        st.write("API Error:", result)