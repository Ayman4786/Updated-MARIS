# Multimodal RAG EdTech Assistant

A multimodal Retrieval-Augmented Generation (RAG) based EdTech assistant that allows users to upload educational documents and ask questions about their content.

The system retrieves relevant document content and, when the user's question requires visual understanding, can also provide the relevant extracted image to the multimodal LLM.

---

## Features

- Upload educational PDFs
- Extract text from uploaded documents
- Extract images/figures from PDFs
- Store extracted images locally
- Split documents into searchable chunks
- Generate embeddings for document chunks
- Hybrid retrieval of relevant content
- Context-aware answer generation
- Multimodal image understanding
- Image retrieval only when required by the user's question
- REST API using FastAPI
- Interactive API testing through Swagger UI

---

## Project Architecture

```text
User
 │
 ▼
Frontend
 │
 ▼
FastAPI Backend
 │
 ├── Upload API
 │      │
 │      ├── PDF Extraction
 │      ├── Text Extraction
 │      └── Image Extraction
 │
 └── Chat API
        │
        ▼
     RAG Pipeline
        │
        ├── Document Chunks
        │
        ├── Embeddings
        │
        ├── Vector Store
        │
        ├── Hybrid Retriever
        │
        └── Context Builder
                │
                ▼
             LLM
                │
                ├── Text-only question
                │      └── Text context
                │
                └── Image-related question
                       ├── Text context
                       └── Relevant image