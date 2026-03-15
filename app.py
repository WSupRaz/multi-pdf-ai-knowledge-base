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

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

PDF_FOLDER = "pdfs"

all_chunks = []

# Read all PDFs
for file in os.listdir(PDF_FOLDER):

    if file.endswith(".pdf"):

        reader = PdfReader(os.path.join(PDF_FOLDER, file))

        text = ""

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text

        # Split into chunks
        chunk_size = 300
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

        all_chunks.extend(chunks)

print("Total chunks:", len(all_chunks))

# Create embeddings
embeddings = model.encode(all_chunks)

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings))

print("FAISS index ready")

# Ask question
question = input("Ask question: ")

q_embedding = model.encode([question])

k = 3

distances, indices = index.search(np.array(q_embedding), k)

top_chunks = [all_chunks[i] for i in indices[0]]

context = "\n".join(top_chunks)

# Send to LLM
data = {
    "model": "openai/gpt-3.5-turbo",
    "messages": [
        {"role": "system", "content": "Answer using only the provided context."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ]
}

response = requests.post(url, headers=headers, json=data)

result = response.json()

print("API response:", result)

if "choices" in result:
    answer = result["choices"][0]["message"]["content"]
    print("\nAI Answer:\n", answer)
else:
    print("API Error:", result)