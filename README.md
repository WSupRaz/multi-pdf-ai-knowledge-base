# Multi-PDF AI Document Assistant

A Retrieval-Augmented Generation (RAG) web app that allows users to upload multiple PDF documents and ask questions about them.

The system extracts text, converts it into embeddings, stores them in a FAISS vector database, and retrieves relevant chunks to generate AI answers.

## Features

- Upload multiple PDFs
- Semantic search using FAISS
- Embedding model (Sentence Transformers)
- AI answers using OpenRouter LLM
- Streamlit chat interface

## Tech Stack

Python  
Streamlit  
FAISS  
Sentence Transformers  
PyPDF2  
OpenRouter API

## Architecture

PDF Upload
↓
Text Chunking
↓
Embeddings
↓
FAISS Vector Database
↓
Semantic Retrieval
↓
LLM Answer

## Demo

Streamlit App: (https://multi-pdf-ai-knowledge-base-fj9ewhiyxcjhmbhchg9xk8.streamlit.app/)
