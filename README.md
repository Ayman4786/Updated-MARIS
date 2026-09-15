# MARIS

## Overview

MARIS is a FastAPI-based document question-answering prototype. It accepts uploaded PDFs, extracts page-aware Markdown and visual elements with Docling, indexes page chunks in a local ChromaDB collection, and answers questions with a hybrid semantic and lexical retriever. When a retrieved chunk has an associated image, the image can be sent to the Groq-hosted Qwen multimodal model together with the retrieved text.

The Python backend is the implemented application surface. The `frontend/` directory is currently the default Vite/React starter screen; it does not yet upload files or call the backend API.

## Implemented Features

- PDF upload through FastAPI
- Page-aware Markdown extraction with Docling
- PDF picture extraction, local PNG storage, captions, page numbers, and bounding-box metadata
- Recursive page chunking with approximately 500-token chunks and 100-token overlap settings
- BGE-small-en-v1.5 text embeddings
- Persistent local ChromaDB storage
- Hybrid retrieval using Chroma vector search and BM25 lexical search
- Optional filtering to one uploaded document
- Text context containing source filename and page metadata
- CLIP-based visual retrieval fallback for explicit visual questions when a document is selected
- Multimodal Groq requests to `qwen/qwen3.6-27b`
- Source and image paths in chat responses
- FastAPI's generated Swagger UI at `/docs`

## Architecture

```text
Client or API tool
            |
            +--> POST /upload
            |       |
            |       +--> Save PDF under storage/documents/<document_id>/
            |       +--> Docling page Markdown and PictureItem extraction
            |       +--> Save document.md, pages.json, visual_manifest.json, images/
            |       +--> Chunk each non-empty page
            |       +--> Embed chunks with BAAI/bge-small-en-v1.5
            |       +--> Upsert chunks and metadata into ChromaDB
            |
            +--> POST /chat
                        |
                        +--> Load chunks from every document's chunks.json
                        +--> Optional document_id filter
                        +--> Chroma semantic search + BM25 lexical search
                        +--> Combine scores and attach nearby page images
                        +--> Optional CLIP visual retrieval fallback
                        +--> Build source/page context and prompt
                        +--> Groq API: qwen/qwen3.6-27b
                        +--> Return answer, sources, and images_used
```

## Project Structure

```text
MARIS/
├── ai/
│   ├── extraction/
│   │   └── docling_extractor.py
│   ├── llm/
│   │   ├── llm_service.py
│   │   ├── prompt_builder.py
│   │   └── question_service.py
│   └── rag/
│       ├── chunk_storage.py
│       ├── chunker.py
│       ├── context_builder.py
│       ├── embeddings_service.py
│       ├── retriever.py
│       ├── vector_store.py
│       └── visual_rag.py
├── backend/
│   ├── main.py
│   ├── config.py
│   ├── routes/
│   │   ├── chat.py
│   │   └── upload.py
│   └── schemas/
├── frontend/
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── App.jsx
│       ├── App.css
│       ├── index.css
│       └── main.jsx
├── storage/
│   └── documents/        # generated per-document artifacts
├── tests/
├── .env.example
├── requirements.txt
└── README.md
```

The repository also contains `storage/chroma_db/` and uploaded-document output in local workspaces. These are runtime data, not source files, and are excluded from the structure above.

## Technology Stack

| Area | Implementation |
| --- | --- |
| Backend API | FastAPI, Uvicorn, Pydantic |
| PDF extraction | Docling and `docling-core` |
| Text embeddings | Hugging Face Transformers, `BAAI/bge-small-en-v1.5`, PyTorch |
| Text retrieval | ChromaDB and `rank-bm25` |
| Visual retrieval | Hugging Face `openai/clip-vit-base-patch32`, PyTorch, Pillow |
| LLM | Groq API with `qwen/qwen3.6-27b` |
| Frontend | React 19 and Vite 8 starter application |

## How It Works

### 1. Document Upload

`POST /upload` accepts a multipart file upload. The backend creates an ID using the original filename stem plus an eight-character UUID suffix, then saves the PDF in `storage/documents/<document_id>/`.

### 2. Document Processing

`ai/extraction/docling_extractor.py` configures Docling to convert the PDF page by page. It uses the `<!-- image -->` placeholder for pictures and enables generated picture images at a scale of 2.0.

### 3. Text and Visual Extraction

The extractor writes:

- `document.md`: combined page Markdown with page markers
- `pages.json`: page character counts and image paths
- `visual_manifest.json`: visual IDs, paths, page numbers, captions, types, and bounding boxes
- `images/image_<n>.png`: extracted pictures

### 4. Chunking and Embeddings

Each non-empty page is split by `RecursiveChunker`. Its defaults are 500 tokens, 100-token overlap, and an approximate four-characters-per-token conversion. Each chunk receives document ID, original filename, chunk ID, and PDF page number metadata. `EmbeddingsService` generates normalized CLS embeddings with `BAAI/bge-small-en-v1.5`, truncating input to 512 model tokens.

### 5. Vector Storage

`VectorStoreManager` uses a persistent ChromaDB client at `./storage/chroma_db` and the `pdf_chunks` collection. Chunk IDs are deterministic MD5 hashes of document ID, filename, chunk index, and chunk text, so uploads use Chroma upserts.

### 6. Retrieval

For chat, the backend loads every document directory containing `chunks.json`. It computes:

1. Chroma semantic results from the question embedding.
2. BM25 lexical results over the loaded chunk text.
3. A combined score of 60% vector score and 40% normalized BM25 score.

The default result limit is five chunks. A supplied `document_id` filters both the in-memory chunks and the Chroma query. Nearby chunks from the same document can contribute an associated image path.

### 7. Visual Retrieval and LLM Response

The chat route recognizes visual questions using explicit visual words and actions such as `explain the diagram` or `describe the chart`. Retrieved images are preferred automatically. If no retrieved chunk has an image and a visual question was detected, `VisualRAG` can compare the question with manifest images using CLIP image similarity and caption similarity. Its current route call supplies the selected document ID, so this fallback is document-specific.

The prompt includes the retrieved text and source/page labels. `llm_service.py` sends the prompt and any existing images as text plus base64 `image_url` parts to Groq. The configured model is `qwen/qwen3.6-27b`, with reasoning hidden and a 1,000-token completion limit.

### 8. Response

`POST /chat` returns the question, generated answer, the selected document ID or `auto`, deduplicated source records, and image paths used by the LLM. Sources include document ID, filename, page, and retrieval score.

## Prerequisites

- Python with a virtual-environment-capable installation. The repository does not declare a required Python version.
- Node.js and npm for the Vite frontend. The repository does not declare required Node.js versions.
- A Groq API key.
- Internet access on first use so Transformers can download the BGE model and, when visual fallback is used, the CLIP model.
- Sufficient local disk space for model caches, extracted images, document artifacts, and ChromaDB data.

No Ollama, MongoDB, PostgreSQL, Docker service, or separately running ChromaDB server is configured or required by the source. ChromaDB runs as a local persistent client.

## Installation

### Backend Setup

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` is a pinned dependency export and includes the packages used by the API, Docling, ChromaDB, Transformers, PyTorch, BM25, and Groq client.

### Frontend Setup

```powershell
Set-Location frontend
npm install
```

## Environment Variables

Create a root `.env` file based on `.env.example`:

```dotenv
GROQ_API_KEY=your_groq_api_key_here
```

`ai/llm/llm_service.py` loads this value with `python-dotenv` and passes it to the Groq client. Do not commit the `.env` file or a real key.

## Running the Application

### Start the Backend

From the repository root, with the virtual environment activated:

```powershell
python -m uvicorn backend.main:app --reload
```

The backend normally listens on `http://127.0.0.1:8000`. Swagger UI is available at `http://127.0.0.1:8000/docs`.

### Start the Frontend

In a second terminal:

```powershell
Set-Location frontend
npm run dev
```

Vite prints the actual local URL, normally `http://localhost:5173`. The current frontend is a Vite demonstration page and has no configured backend proxy or API calls.

## API Endpoints

### `GET /`

Returns:

```json
{"message": "Server Running"}
```

### `POST /upload`

Accepts a multipart/form-data field named `file`:

```text
file=<PDF file>
```

The response reports the original filename, generated document ID, status, page/chunk/embedding/image counts, output paths, and the first 1,000 characters of extracted Markdown. Processing is synchronous and performs extraction, embedding, and ChromaDB upsert before returning.

### `POST /chat`

Accepts JSON:

```json
{
   "question": "What does the diagram show?",
   "document_id": "optional_document_directory_name"
}
```

`document_id` is optional. When omitted, all uploaded documents with valid `chunks.json` files are searched. When supplied, it must match an uploaded document directory name. The response contains `question`, `document_id`, `answer`, `sources`, and `images_used`.

The routes in `backend/routes/highlight.py` and `backend/routes/session.py` are currently empty and are not registered. The files under `backend/schemas/` are also empty; the upload and chat routes currently define or use their request data directly.

## Data and RAG Workflow

```text
PDF upload
   -> storage/documents/<id>/<original-file>
   -> Docling page Markdown + PictureItem images
   -> document.md, pages.json, visual_manifest.json
   -> page chunks with document/page/image metadata
   -> BGE embeddings
   -> ChromaDB pdf_chunks collection

Question
   -> load chunks.json files
   -> optional document filter
   -> Chroma semantic search + BM25 lexical search
   -> weighted ranking and same-document image association
   -> optional CLIP visual retrieval
   -> source/page context and prompt
   -> Groq Qwen response
   -> answer, sources, and image paths
```

## Testing

The repository contains exploratory Python scripts rather than a configured test runner. Some scripts load models, write local storage, or reference paths that are not part of the current generated storage layout. There is no `pytest` configuration in the repository.

The frontend package defines these checks:

```powershell
Set-Location frontend
npm run lint
npm run build
```

The backend dependency set includes `pytest`-related tooling only indirectly; `pytest` itself is not declared in `requirements.txt`. Run backend test scripts only after checking their input paths and model/storage prerequisites.

## Troubleshooting

- **Missing Groq key:** create `.env` at the repository root with `GROQ_API_KEY`. The LLM client is initialized from that value.
- **First request is slow or fails while offline:** BGE and CLIP model files are downloaded by Transformers and must be available locally.
- **No documents found:** upload a file first and confirm that `storage/documents/<document_id>/chunks.json` was created successfully.
- **Requested document not found:** use the exact `document_id` returned by `/upload`, not the original filename alone.
- **Images are not used:** only existing image paths attached to retrieved chunks are sent automatically. Visual fallback also requires a valid visual manifest and a selected document ID.
- **Frontend does not upload or chat:** this is expected for the current Vite starter implementation; use the API directly through Swagger UI or an HTTP client.
- **Large or rate-limited LLM requests:** the backend returns a user-facing message for Groq authentication, rate-limit, oversized-request, and general API errors.

## Future Improvements

These are not currently implemented:

- Connect the React frontend to `/upload` and `/chat`.
- Add document listing, chat history, highlight, and session APIs.
- Add automated backend tests with isolated temporary storage and mocked model/API calls.
- Add explicit Python and Node version files or documentation.
- Add configurable storage paths, model names, retrieval limits, and server settings.
- Add cleanup, file validation, and lifecycle management for uploaded documents.

## License

No license file or license declaration is present in the repository.