# Multi-PDF AI Knowledge Base

This project implements a Retrieval-Augmented Generation (RAG) system that allows users to ask questions across multiple PDF documents.

## Features

- Reads multiple PDFs
- Splits text into chunks
- Converts text into embeddings using Sentence Transformers
- Stores embeddings in FAISS vector database
- Performs semantic search
- Uses LLM to generate answers from document context

## Tech Stack

- Python
- FAISS
- Sentence Transformers
- PyPDF2
- OpenRouter API

## Architecture

PDFs → Chunking → Embeddings → FAISS → Semantic Search → LLM Answer

## Run Locally
