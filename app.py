#dual thread tested on 15-06-26 works well cant parse tables too well and have problems with similairity ranking
#as of now uses chromadb for embedding and retrieval, uses google gemini 2.5 flash for llm, uses docling for pdf parsing and image extraction and langchain for text splitting and llm interface, uses rank-bm25 for bm25 scoring and rrf for hybrid ranking
import os
import io
import json
import shutil
import time
import traceback
import logging
import re
import threading
import math


logging.basicConfig(level=logging.INFO)
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup Directories
BASE_DIR = Path(__file__).resolve().parent
DATA_FOLDER = BASE_DIR / "data"
IMAGES_FOLDER = BASE_DIR / "extracted_images"
CHROMA_DIR = BASE_DIR / "chromadb"

DATA_FOLDER.mkdir(exist_ok=True)
IMAGES_FOLDER.mkdir(exist_ok=True)

# Initialize FastAPI App
app = FastAPI(title="Docling RAG System")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup ChromaDB
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = chroma_client.get_or_create_collection(name="documents")

# ─────────────────────────────────────────────
# Job Tracking — stores status of every upload
# ─────────────────────────────────────────────
# Each job looks like:
# {
#   "status": "processing" | "complete" | "failed",
#   "filename": "myfile.pdf",
#   "progress": "Running Docling conversion...",
#   "chunks_added": 0,
#   "total_chunks": 0,
#   "error": ""           (only on failure)
# }
processing_jobs = {}

# Lazy load LLM and Docling to speed up app startup
llm = None
converter = None

def get_llm():
    global llm
    if llm is not None:
        return llm

    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        print("[WARNING] GOOGLE_API_KEY environment variable not set.")
        return None

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            max_retries=3,
            request_timeout=30
        )
        print("[OK] Google Gemini 2.5 Flash initialized successfully.")
        return llm
    except Exception as e:
        print(f"[ERROR] Failed to initialize Gemini: {e}")
        return None


def get_converter(images_scale: float = 2.0, generate_images: bool = True):
    """
    Build a fresh Docling converter with the given settings.
    NOT cached globally because settings change based on file size.
    """
    try:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.base_models import InputFormat

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        pipeline_options.generate_picture_images = generate_images
        pipeline_options.images_scale = images_scale

        conv = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )
        logging.info(f"[OK] Docling converter ready — scale={images_scale}, images={generate_images}")
        return conv
    except Exception as e:
        logging.error(f"[ERROR] Failed to initialize Docling: {e}")
        raise RuntimeError(f"Failed to initialize Docling parser: {str(e)}")


# Cache setup
CACHE_FILE = BASE_DIR / "query_cache.json"
query_cache = {}
if CACHE_FILE.exists():
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            query_cache = json.load(f)
    except Exception as e:
        print(f"Failed to load cache: {e}")

last_api_call = {"time": 0}


def save_cache():
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(query_cache, f, indent=2)
    except Exception as e:
        print(f"Failed to save cache: {e}")


# ─────────────────────────────────────────────
# Request / Response Models
# ─────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    use_cache: bool = True
    n_results: int = 10
    chunk_chars: int = 1000


class QueryResponse(BaseModel):
    query: str
    answer: str
    cached: bool
    sources: List[dict]
    image_paths: List[str] = []


class StatusResponse(BaseModel):
    chunk_count: int
    documents: List[str]
    gemini_active: bool
    api_key_configured: bool


class JobStatusResponse(BaseModel):
    job_id: str
    status: str          # "processing" | "complete" | "failed"
    filename: str
    progress: str
    chunks_added: int
    total_chunks: int
    error: str


# ─────────────────────────────────────────────
# Background Processing Function
# Runs in a separate thread — NEVER blocks FastAPI
# ─────────────────────────────────────────────

def process_pdf_background(job_id: str, file_path: Path, filename: str,
                            images_scale: float, generate_images: bool):
    """
    All the heavy Docling + ChromaDB work happens here,
    in a background thread completely separate from the main server thread.
    If this freezes or crashes, FastAPI keeps running normally.
    """

    def update(progress: str):
        processing_jobs[job_id]["progress"] = progress
        logging.info(f"[Job {job_id}] {progress}")

    try:
        update("Starting Docling conversion...")

        # ── Docling conversion ──────────────────────────────────────────
        conv = get_converter(images_scale=images_scale, generate_images=generate_images)
        result = conv.convert(str(file_path))
        doc = result.document
        update("Docling conversion complete. Exporting Markdown...")

        # ── Markdown export ─────────────────────────────────────────────
        markdown_content = doc.export_to_markdown()
        md_path = DATA_FOLDER / f"{file_path.stem}_output.md"
        with md_path.open("w", encoding="utf-8") as f:
            f.write(markdown_content)
        update("Markdown saved. Extracting image contexts...")

        # ── Image context extraction from Markdown ──────────────────────
        image_contexts_from_markdown = []
        matches = re.findall(r'([^\n]+?)\s*<!-- image -->', markdown_content, re.DOTALL)
        for match_text in matches:
            ctx = match_text.strip()
            ctx = re.sub(r'#+\s*', '', ctx)
            ctx = re.sub(r'\s*\|.*', '', ctx)
            ctx = ctx.replace('\\', '').strip()
            image_contexts_from_markdown.append(ctx)

        # ── Image extraction ────────────────────────────────────────────
        image_docs_to_index = []
        ts = int(time.time())

        if generate_images:
            update(f"Extracting images (found {len(doc.pictures)})...")
            for i, picture in enumerate(doc.pictures):
                try:
                    pil_img = picture.image.pil_image
                    if pil_img is None:
                        continue
                    page = picture.prov[0].page_no if picture.prov else "unknown"
                    img_filename = f"{file_path.stem}_page{page}_img{i}.png"
                    img_path = IMAGES_FOLDER / img_filename
                    pil_img.save(img_path)

                    ocr_text = ""
                    if hasattr(picture, 'ocr_text') and picture.ocr_text is not None:
                        ocr_obj = picture.ocr_text
                        if hasattr(ocr_obj, 'text') and ocr_obj.text is not None:
                            ocr_text = ocr_obj.text

                    markdown_context = ""
                    if i < len(image_contexts_from_markdown):
                        markdown_context = image_contexts_from_markdown[i]

                    image_document_content = f"Image {i} from page {page}. "
                    if ocr_text:
                        image_document_content += f"OCR text: {ocr_text}. "
                    elif markdown_context:
                        image_document_content += f"Context from document: {markdown_context}. "
                    else:
                        image_document_content += "No readable text or explicit context found in this image. "

                    db_img_path = f"extracted_images/{img_filename}"
                    image_document_content += f"File: {db_img_path}"

                    image_docs_to_index.append({
                        "id": f"{filename}_image_{i}_{page}_{ts}",
                        "document": image_document_content,
                        "metadata": {
                            "source": filename,
                            "type": "image",
                            "page": page,
                            "image_path": db_img_path,
                            "ocr_text_present": int(bool(ocr_text)),
                            "markdown_context_present": int(bool(markdown_context)),
                            "chunk_id": f"image_{i}"
                        }
                    })
                except Exception as img_err:
                    logging.warning(f"[Job {job_id}] Failed to save image {i}: {img_err}")
        else:
            update("Image extraction skipped (large file mode).")

        # ── Text chunking ───────────────────────────────────────────────
        update("Chunking Markdown text...")
        from langchain_text_splitters import MarkdownTextSplitter
        splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(markdown_content)
        logging.info(f"[Job {job_id}] Generated {len(chunks)} text chunks.")

        if not chunks and not image_docs_to_index:
            raise RuntimeError("No content (text or images) could be extracted from this PDF.")

        # ── ChromaDB indexing ───────────────────────────────────────────
        total_added = 0

        if chunks:
            update(f"Indexing {len(chunks)} text chunks into ChromaDB...")
            chunk_ids = [f"{filename}_chunk_{i}_{ts}" for i in range(len(chunks))]
            collection.add(
                ids=chunk_ids,
                documents=chunks,
                metadatas=[{"source": filename, "chunk_id": i, "type": "text"} for i in range(len(chunks))]
            )
            total_added += len(chunks)

        if image_docs_to_index:
            update(f"Indexing {len(image_docs_to_index)} image documents into ChromaDB...")
            collection.add(
                ids=[d["id"] for d in image_docs_to_index],
                documents=[d["document"] for d in image_docs_to_index],
                metadatas=[d["metadata"] for d in image_docs_to_index]
            )
            total_added += len(image_docs_to_index)

        # ── Mark job complete ───────────────────────────────────────────
        processing_jobs[job_id].update({
            "status": "complete",
            "progress": "Processing complete.",
            "chunks_added": total_added,
            "total_chunks": collection.count()
        })
        logging.info(f"[Job {job_id}] Complete. {total_added} items indexed.")

    except Exception as e:
        tb = traceback.format_exc()
        logging.error(f"[Job {job_id}] FAILED:\n{tb}")

        # Clean up orphaned file on failure
        try:
            if file_path.exists():
                file_path.unlink()
                logging.info(f"[Job {job_id}] Cleaned up orphaned file {file_path.name}")
        except Exception as cleanup_err:
            logging.warning(f"[Job {job_id}] Could not clean up file: {cleanup_err}")

        processing_jobs[job_id].update({
            "status": "failed",
            "progress": "Processing failed.",
            "error": str(e)
        })


# ─────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────

@app.get("/api/status", response_model=StatusResponse)
def get_status(): 
#how many chunks loaded, how many documents, is api key active, is llm working
    count = collection.count()#chunks
    documents = set()#to prevent documents from repeating
    if count > 0:
        try:
            results = collection.get(include=["metadatas"])
            if results and results.get("metadatas"):
                for meta in results["metadatas"]:
                    if meta and "source" in meta:
                        documents.add(meta["source"])
        except Exception as e:
            print(f"Error fetching metadata: {e}")

    api_key = os.getenv("GOOGLE_API_KEY", "")
    has_api_key = len(api_key.strip()) > 0

    return StatusResponse(
        chunk_count=count,
        documents=list(documents),
        gemini_active=(get_llm() is not None),
        api_key_configured=has_api_key
    )


@app.get("/api/job/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str):
    """
    Poll this endpoint to check the progress of a background upload.
    Frontend calls this every few seconds after receiving a job_id from /api/upload.
    """
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    job = processing_jobs[job_id]
    return JobStatusResponse(
        job_id=job_id,
        status=job.get("status", "unknown"),
        filename=job.get("filename", ""),
        progress=job.get("progress", ""),
        chunks_added=job.get("chunks_added", 0),
        total_chunks=job.get("total_chunks", 0),
        error=job.get("error", "")
    )


@app.get("/api/jobs")
def list_jobs():
    """Returns all jobs — useful for debugging."""
    return processing_jobs


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    1. Validates file type
    2. Checks page count (rejects if over limit)
    3. Saves file to disk
    4. Determines memory-safe Docling settings based on file size
    5. Starts background thread for all heavy processing
    6. Returns job_id IMMEDIATELY — server never freezes
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    filename = file.filename
    file_path = DATA_FOLDER / filename #after checking the file type, save it to folder (data)

    # ── Read file bytes ─────────────────────────────────────────────────
    contents = await file.read()
    file_size_mb = len(contents) / (1024 * 1024)
    logging.info(f"Received {filename} — {file_size_mb:.1f} MB")

    # ── Page count guard ─────────────────────────────────────────────────
    #not needed, takes alot of load.
    # Uses pypdf (very lightweight) to count pages BEFORE Docling loads
    # This check costs almost zero memory and completes in milliseconds
    #page_count = None
    #try:
    #    import pypdf
    #    reader = pypdf.PdfReader(io.BytesIO(contents)) #reads it instead of loading the whole file into memory
    #    page_count = len(reader.pages)
    #    logging.info(f"{filename} has {page_count} pages.")
    #    if page_count > 100:
    #        raise HTTPException(
    #            status_code=400,
    #            detail=(
    #                f"PDF too large: {page_count} pages. "
    #                f"Maximum allowed is 100 pages. "
    #                f"Please split the document and upload in parts."
    #            )
    #        )
    #except HTTPException:
    #    raise
    #except Exception as e:
        # pypdf failed to read — let Docling try anyway
    #    logging.warning(f"Could not count pages with pypdf: {e}")

    # ── Save file to disk ────────────────────────────────────────────────
    with file_path.open("wb") as buffer:
        buffer.write(contents)
    logging.info(f"Saved {filename} to {file_path}")

    # ── Determine memory-safe Docling settings ───────────────────────────
    #
    # File size  | images_scale | generate_images | Why
    # -----------+--------------+-----------------+-------------------------
    # < 10 MB    |     2.0      |      True       | Small file, full quality
    # 10-20 MB   |     1.0      |      True       | Medium, 75% less memory
    # > 20 MB    |     1.0      |      False      | Large, images disabled
    #
    if file_size_mb < 10:
        images_scale = 2.0
        generate_images = True
        mode = "full quality"
    elif file_size_mb < 20:
        images_scale = 1.0
        generate_images = True
        mode = "reduced quality (medium file)"
    else:
        images_scale = 1.0
        generate_images = False
        mode = "text only (large file)"

    logging.info(f"Processing mode: {mode} — scale={images_scale}, images={generate_images}")

    # ── Create job entry ─────────────────────────────────────────────────
    job_id = f"job_{filename.replace('.', '_')}_{int(time.time())}"
    processing_jobs[job_id] = {
        "status": "processing",
        "filename": filename,
        "progress": "Upload received. Starting background processing...",
        "chunks_added": 0,
        "total_chunks": collection.count(),
        "error": "",
        "file_size_mb": round(file_size_mb, 1),
        "page_count":None,
        "mode": mode
    }

    # ── Start background thread ──────────────────────────────────────────
    # Docling runs here — completely separate from FastAPI's main thread
    # Even if this thread freezes, the web server stays responsive
    thread = threading.Thread(
        target=process_pdf_background, #this fucntion runs in the background
        args=(job_id, file_path, filename, images_scale, generate_images),
        daemon=True  # thread dies automatically if main process exits
    )
    thread.start()

    # ── Return immediately ───────────────────────────────────────────────
    # User gets a response in milliseconds
    # Frontend polls /api/job/{job_id} to track progress
    return {
        "status": "processing",
        "job_id": job_id,
        "filename": filename,
        "file_size_mb": round(file_size_mb, 1),
        "page_count": None,
        "processing_mode": mode,
        "message": "File received. Processing started in background.",
        "poll_url": f"/api/job/{job_id}"
    }


@app.post("/api/clear-docs")
def clear_documents():
    try:
        for item in DATA_FOLDER.glob("*"):
            if item.is_file():
                item.unlink()
        for item in IMAGES_FOLDER.glob("*"):
            if item.is_file():
                item.unlink()
        display_md = BASE_DIR / "document_output.md"
        if display_md.exists():
            display_md.unlink()
        return {"status": "success", "message": "Raw documents and extracted data cleared from disk."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear files: {str(e)}")


@app.post("/api/clear-chroma")
def clear_chroma():
    global query_cache
    try:
        existing = collection.get()
        deleted_count = 0
        if existing and existing["ids"]:
            deleted_count = len(existing["ids"])
            collection.delete(ids=existing["ids"])
        query_cache = {}
        if CACHE_FILE.exists():
            CACHE_FILE.unlink()
        return {
            "status": "success",
            "message": f"Successfully deleted {deleted_count} chunks from ChromaDB and cleared query cache."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear ChromaDB: {str(e)}")


# ─────────────────────────────────────────────
# BM25 + RRF Hybrid Retrieval Helpers
# ─────────────────────────────────────────────

def tokenize_for_bm25(text: str) -> List[str]:
    """Lowercase, split on non-alphanumeric. Fast and dependency-light."""
    return re.findall(r'\w+', text.lower())#helps in saving in lowercase


def reciprocal_rank_fusion(rankings: List[List[int]], k: int = 60) -> List[tuple]:
    """
    Merge multiple ranked lists (each is a list of doc indices) using RRF.
    Returns [(doc_index, score), ...] sorted by score descending.
    k=60 is the standard smoothing constant from the original RRF paper.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_idx in enumerate(ranking):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def bm25_rerank(query: str, documents: List[str]) -> List[int]:
    """
    Score `documents` against `query` using BM25.
    Returns indices sorted best-first.
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        # rank-bm25 not installed — fall back to original order
        logging.warning("rank-bm25 not installed. Run: pip install rank-bm25. Falling back to semantic order.")
        return list(range(len(documents)))

    tokenized_corpus = [tokenize_for_bm25(doc) for doc in documents]
    tokenized_query = tokenize_for_bm25(query)

    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenized_query)
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)


# ─────────────────────────────────────────────
# Query Endpoint — Hybrid BM25 + Semantic RAG
# ─────────────────────────────────────────────

@app.post("/api/query", response_model=QueryResponse)
def query_rag_api(req: QueryRequest):
    # ── Cache check ──────────────────────────────────────────────────────
    if req.use_cache and req.query in query_cache:
        cached_val = query_cache[req.query]
        if isinstance(cached_val, dict):
            return QueryResponse(
                query=req.query,
                answer=cached_val.get("answer", ""),
                cached=True,
                sources=cached_val.get("sources", []),
                image_paths=cached_val.get("image_paths", [])
            )
        else:
            # Legacy string-only cache entry
            return QueryResponse(
                query=req.query,
                answer=cached_val,
                cached=True,
                sources=[],
                image_paths=[]
            )

    llm_instance = get_llm()
    if llm_instance is None:
        raise HTTPException(status_code=400, detail="Gemini LLM is not configured. Please set GOOGLE_API_KEY.")

    count = collection.count()
    if count == 0:
        raise HTTPException(status_code=400, detail="No documents indexed in ChromaDB. Please upload a PDF first.")

    # ── Rate limiting ────────────────────────────────────────────────────
    time_since_last_call = time.time() - last_api_call["time"]
    min_delay = 4.0
    if time_since_last_call < min_delay:
        time.sleep(min_delay - time_since_last_call)

    try:
        # ── Step 1: Semantic retrieval (wider candidate pool for reranking) ──
        # Fetch 3x candidates so BM25 has enough to rerank meaningfully.
        # Final result is still trimmed to req.n_results.
        candidate_multiplier = 3
        n_candidates = min(req.n_results * candidate_multiplier, count) #capped at total count
#what embeds the query, gets similar results with doc and metadata
        chroma_res = collection.query(
            query_texts=[req.query],
            n_results=n_candidates,
            include=["documents", "metadatas"]
        )

        if not chroma_res or not chroma_res["documents"] or not chroma_res["documents"][0]:
            return QueryResponse(
                query=req.query,
                answer="No relevant content found in the database to answer this question.",
                cached=False,
                sources=[],
                image_paths=[]
            )

        candidate_docs = chroma_res["documents"][0]
        candidate_metas = chroma_res["metadatas"][0]

        # ── Step 2: BM25 reranking over the candidate pool ──────────────
        # semantic_order  = [0, 1, 2, ...] — ChromaDB already returns best-first
        # bm25_order      = indices sorted by BM25 score
       
        semantic_order = list(range(len(candidate_docs)))
        bm25_order = bm25_rerank(req.query, candidate_docs)
        fused = reciprocal_rank_fusion([semantic_order, bm25_order]) # RRF blends both to produce final ranking

        # Take top n_results after fusion
        top_indices = [idx for idx, _score in fused[:req.n_results]]

        retrieved_docs = [candidate_docs[i] for i in top_indices]
        retrieved_metas = [candidate_metas[i] for i in top_indices]

        logging.info(
            f"[Query] '{req.query[:60]}' — "
            f"{len(candidate_docs)} candidates → {len(retrieved_docs)} after BM25+RRF"
        )

        # ── Step 3: Build sources + image list ───────────────────────────
        sources = []
        image_paths = []
        for doc, meta in zip(retrieved_docs, retrieved_metas):
            is_image = meta.get("type") == "image"
            img_path = meta.get("image_path", "")
            if is_image and img_path:
                url_path = f"/{img_path}" if not img_path.startswith("/") else img_path
                image_paths.append(url_path)
                sources.append({
                    "source": meta.get("source", "unknown"),
                    "chunk_id": meta.get("chunk_id", -1),
                    "type": "image",
                    "image_path": url_path,
                    "snippet": doc[:req.chunk_chars]
                })
            else:
                sources.append({
                    "source": meta.get("source", "unknown"),
                    "chunk_id": meta.get("chunk_id", -1),
                    "type": "text",
                    "snippet": doc[:req.chunk_chars]
                })

        # ── Step 4: LLM call ─────────────────────────────────────────────
        context = "\n\n---\n\n".join(
            f"Source: {meta.get('source', 'unknown')}, Type: {meta.get('type', 'text')}\n{doc[:req.chunk_chars]}"
            for doc, meta in zip(retrieved_docs, retrieved_metas)
        )
        prompt = f"Answer based on this context only.\nQuery: {req.query}\nContext: {context}\nAnswer:"

        response = llm_instance.invoke(prompt)
        answer = response.content if hasattr(response, 'content') else str(response)

        last_api_call["time"] = time.time()
        query_cache[req.query] = {
            "answer": answer,
            "sources": sources,
            "image_paths": image_paths
        }
        save_cache()

        return QueryResponse(
            query=req.query,
            answer=answer,
            cached=False,
            sources=sources,
            image_paths=image_paths
        )

    except Exception as e:
        logging.error(f"Error querying RAG: {e}")
        raise HTTPException(status_code=500, detail=f"Error executing RAG query: {str(e)}")

# Mount Static Files (at the end so API routes take precedence)
static_dir = BASE_DIR / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/extracted_images", StaticFiles(directory=str(IMAGES_FOLDER)), name="images")
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)

