# Document Parsing and Cross Conversation Project

## Introduction
In todays day and age documents have become an integral part of our professional and academic life, to some extent even personal! From important books/ material to offical reports everything have majorly shifted to online pdf documents. 

These documents are generally in pdf format. They are portable, easy to access and use. But they can be bulky. For example if you are going through a book on Machine Learning, or going through an annual report for your company--they all require undivided long term attention to be understood and read properly. Even if you're accessing something for a specific topic/ field, it would still take time to find the correct context and material in the document to properly understand it. Apart from this the material may not be easy to understand and may require multiple reads to get it. 

# Docling Hybrid RAG System

A high-performance Retrieval-Augmented Generation (RAG) backend built with FastAPI. This system handles complex PDF documents (including embedded tables and pictures) using Docling, performs hybrid retrieval combining Semantic Search (ChromaDB) and Keyword Search (BM25), and generates answers using Google Gemini 2.5 Flash.

---

## 🚀 Key Features

- **Advanced Document Parsing:** Uses Docling for precise PDF text layout formatting and structured extraction of embedded images.
- **True Asynchronous Background Processing:** File processing happens entirely inside non-blocking background worker threads. FastAPI remains responsive even under heavy parsing loads.
- **Adaptive Memory-Safe Engine:** Automatically scales down picture rendering resolution or falls back to text-only mode depending on file size to prevent Out-Of-Memory (OOM) crashes.
- **Hybrid Retrieval Pipeline:** Blends Dense Semantic Embeddings (ChromaDB) with Sparse Token-matching (BM25) via Reciprocal Rank Fusion (RRF).
- **Atomic Query Caching:** Safely writes query responses to local JSON cache using `threading.Lock` and atomic file system replacement (`os.replace`) to eliminate race conditions.
- **Lazy Initialization:** Both the LLM and Docling converter are loaded on first use, keeping app startup fast.
- **Static File Serving:** Extracted images and a frontend UI are served directly by the FastAPI app via `StaticFiles`.

---

## 🛠️ Architecture Overview

The backend processing flow is split safely between web-server interactions and heavy file parsing tasks:

```
[User Upload] ──> FastAPI Endpoint ──> Saves File & Dispatches Worker Thread
                                                      │
              ┌───────────────────────────────────────┘
              ▼
[Background Thread] ──> Docling Parsing ──> Markdown & Image Extraction
                                                      │
                                                      ▼
                              ┌───────────────────────────────────────┐
                              │         ChromaDB Indexing             │
                              │  (text chunks + image context docs)   │
                              └───────────────────────────────────────┘
```

### Query Flow

```
[POST /api/query]
      │
      ├── Cache hit? ──> Return cached response immediately
      │
      └── Cache miss
            │
            ├── Step 1: ChromaDB semantic search (3× candidate pool)
            ├── Step 2: BM25 reranking over candidates
            ├── Step 3: Reciprocal Rank Fusion (RRF) — merge both rankings
            ├── Step 4: LLM call (Gemini 2.5 Flash)
            └── Step 5: Save result to cache (thread-safe, atomic write)
```

---

## 📋 Tech Stack

| Component | Framework / Technology |
|---|---|
| API Framework | FastAPI + Uvicorn |
| PDF Parser | Docling (IBM) |
| Vector Database | ChromaDB (Cosine similarity space) |
| Keyword Reranking | Rank-bm25 (BM25Okapi) |
| Hybrid Ranking | Reciprocal Rank Fusion (RRF) |
| Text Splitting | LangChain `MarkdownTextSplitter` |
| LLM Orchestrator | LangChain (langchain_google_genai) |
| Inference Engine | Google Gemini 2.5 Flash |
| Image Processing | Pillow (PIL) |

---

## 📁 Directory Structure

```
project-root/
├── app.py
├── .env
├── query_cache.json        # Auto-generated; persists query responses
├── data/                   # Uploaded PDFs + exported Markdown files
├── extracted_images/       # PNG images extracted from PDFs
├── chromadb/               # Persistent ChromaDB vector store
└── static/                 # Frontend UI served at /
```

---

## ⚙️ Setup and Installation

### 1. Clone & Navigate

```bash
git clone <repository-url>
cd <repository-directory>
```

### 2. Configure Environment Variables

Create a `.env` file in the root directory:

```
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 3. Install Dependencies

```bash
pip install fastapi uvicorn chromadb langchain-google-genai docling rank-bm25 pydantic python-dotenv pillow langchain-text-splitters
```

### 4. Run the Application

```bash
python app.py
```

The server will start at `http://127.0.0.1:8000`.

---

## 🔌 API Documentation

### Document Management

#### `POST /api/upload`

Uploads a PDF file. Automatically selects a memory-safe processing mode based on file size, then dispatches a background worker and returns a `job_id` instantly.

- **Payload:** Multipart Form-Data (`file: UploadFile`)
- **Processing Modes (selected automatically):**

| File Size | `images_scale` | `generate_images` | Mode |
|---|---|---|---|
| < 10 MB | 2.0 | Yes | Full quality |
| 10–20 MB | 1.0 | Yes | Reduced quality |
| > 20 MB | 1.0 | No | Text only |

- **Response:**

```json
{
  "status": "processing",
  "job_id": "job_filename_1719660000",
  "filename": "manual.pdf",
  "file_size_mb": 4.2,
  "page_count": null,
  "processing_mode": "full quality",
  "message": "File received. Processing started in background.",
  "poll_url": "/api/job/job_filename_1719660000"
}
```

---

#### `GET /api/job/{job_id}`

Polls the processing status of a background upload job. Call this every few seconds after receiving a `job_id` from `/api/upload`.

- **Statuses:** `"processing"`, `"complete"`, `"failed"`
- **Response:**

```json
{
  "job_id": "job_filename_1719660000",
  "status": "complete",
  "filename": "manual.pdf",
  "progress": "Processing complete.",
  "chunks_added": 42,
  "total_chunks": 42,
  "error": ""
}
```

---

#### `GET /api/jobs`

Returns all tracked jobs. Useful for debugging.

---

#### `GET /api/status`

Returns statistics about the current system state.

- **Response:**

```json
{
  "chunk_count": 120,
  "documents": ["manual.pdf", "report.pdf"],
  "gemini_active": true,
  "api_key_configured": true
}
```

---

#### `POST /api/clear-docs`

Wipes all uploaded PDFs, exported Markdown files, and extracted images from disk. Does **not** touch ChromaDB.

---

#### `POST /api/clear-chroma`

Deletes all indexed chunks from ChromaDB and clears the query cache (`query_cache.json`). Does **not** delete raw files from disk.

---

### Retrieval & Inference

#### `POST /api/query`

Executes hybrid BM25 + semantic search over indexed chunks and queries the Gemini LLM.

- **Request Payload:**

```json
{
  "query": "What are the structural specifications listed in chapter 4?",
  "use_cache": true,
  "n_results": 10,
  "chunk_chars": 1000
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `query` | string | required | The question to answer |
| `use_cache` | bool | `true` | Return cached answer if available |
| `n_results` | int | `10` | Number of final chunks passed to the LLM |
| `chunk_chars` | int | `1000` | Max characters per chunk shown in sources/context |

- **Response:**

```json
{
  "query": "What are the structural specifications listed in chapter 4?",
  "answer": "Based on the provided document context, chapter 4 specifies...",
  "cached": false,
  "sources": [
    {
      "source": "manual.pdf",
      "chunk_id": 14,
      "type": "text",
      "snippet": "..."
    },
    {
      "source": "manual.pdf",
      "chunk_id": "image_2",
      "type": "image",
      "image_path": "/extracted_images/manual_page3_img2.png",
      "snippet": "Image 2 from page 3. Context from document: ..."
    }
  ],
  "image_paths": ["/extracted_images/manual_page3_img2.png"]
}
```

> **Rate limiting:** A minimum 4-second delay is enforced between successive LLM calls to avoid API quota exhaustion.

---

## 🔒 Concurrency and Data Integrity

- **Concurrency Lock:** Multi-threaded updates to `query_cache.json` are isolated using `threading.Lock()` to prevent state corruption.
- **Atomic Write Protocol:** Cache writes use `tempfile.mkstemp` to create a temporary file, `f.flush()` + `os.fsync()` to ensure data is committed to disk, then `os.replace()` to atomically swap in the new file — eliminating partial-write corruption.
- **Daemon Threads:** Background processing threads are started with `daemon=True`, so they are automatically killed if the main process exits.
- **Orphan Cleanup:** If a background job fails mid-way, the partially saved PDF is automatically deleted from disk.

---

## 🖼️ Image Handling Details

Extracted images are downsampled by 50% (via `LANCZOS` resampling) before saving to reduce disk usage. Each image is indexed into ChromaDB as a text document containing:

- Page number and image index
- OCR text (if available via `picture.ocr_text`)
- Surrounding Markdown context (up to 200 characters before the `<!-- image -->` tag)
- File path to the saved PNG

This allows image-related queries to surface relevant figures even without visual embedding models.
