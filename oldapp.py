import os
import io
import json
import shutil
import time
import traceback
import logging
import re

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

def get_converter():
    global converter
    if converter is not None:
        return converter
        
    try:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.base_models import InputFormat

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.generate_picture_images = True
        pipeline_options.images_scale = 2.0

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )
        print("[OK] Docling DocumentConverter initialized.")
        return converter
    except Exception as e:
        print(f"[ERROR] Failed to initialize Docling: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize Docling parser: {str(e)}")

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

# Request/Response Models
class QueryRequest(BaseModel):
    query: str
    use_cache: bool = True
    n_results: int = 5
    chunk_chars: int = 500

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

# API Endpoints
@app.get("/api/status", response_model=StatusResponse)
def get_status():
    # Count chunks in collection
    count = collection.count()
    
    # Get distinct source documents from metadata
    documents = set()
    if count > 0:
        try:
            # Retrieve metadata from collection
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

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    filename = file.filename
    file_path = DATA_FOLDER / filename

    try:
        # Save the file to data/
        contents = await file.read()
        with file_path.open("wb") as buffer:
            buffer.write(contents)
        logging.info(f"Saved {filename} ({len(contents)} bytes) to {file_path}")

        # Convert document with Docling
        logging.info("Initialising Docling converter...")
        conv = get_converter()
        logging.info("Running Docling conversion...")
        result = conv.convert(str(file_path))
        doc = result.document
        logging.info("Docling conversion done.")

        # Export markdown
        markdown_content = doc.export_to_markdown()
        md_path = DATA_FOLDER / f"{file_path.stem}_output.md"
        with md_path.open("w", encoding="utf-8") as f:
            f.write(markdown_content)
        logging.info(f"Markdown saved to {md_path}")

        # Load markdown content to extract image contexts/captions
        image_contexts_from_markdown = []
        matches = re.findall(r'([^\n]+?)\s*<!-- image -->', markdown_content, re.DOTALL)
        for match_text in matches:
            context_candidate = match_text.strip()
            context_candidate = re.sub(r'#+\s*', '', context_candidate) # Remove markdown headers
            context_candidate = re.sub(r'\s*\|.*', '', context_candidate) # Remove table lines
            context_candidate = context_candidate.replace('\\', '').strip() # Remove backslashes
            image_contexts_from_markdown.append(context_candidate)

        # Extract images and prepare for indexing
        image_docs_to_index = []
        ts = int(time.time())
        for i, picture in enumerate(doc.pictures):
            try:
                pil_img = picture.image.pil_image
                if pil_img is None:
                    continue
                page = picture.prov[0].page_no if picture.prov else "unknown"
                img_filename = f"{file_path.stem}_page{page}_img{i}.png"
                img_path = IMAGES_FOLDER / img_filename
                pil_img.save(img_path)
                
                # Prepare image data for ChromaDB
                ocr_text = ""
                if hasattr(picture, 'ocr_text') and picture.ocr_text is not None:
                    ocr_content_obj = picture.ocr_text
                    if hasattr(ocr_content_obj, 'text') and ocr_content_obj.text is not None:
                        ocr_text = ocr_content_obj.text

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
                logging.warning(f"Failed to save image {i}: {img_err}")

        # Chunk the markdown
        from langchain_text_splitters import MarkdownTextSplitter
        splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_text(markdown_content)
        logging.info(f"Generated {len(chunks)} text chunks.")

        if not chunks and not image_docs_to_index:
            raise HTTPException(status_code=400, detail="No content (text or images) extracted from the PDF.")

        # Add text chunks to ChromaDB with unique IDs
        if chunks:
            chunk_ids = [f"{filename}_chunk_{i}_{ts}" for i in range(len(chunks))]
            collection.add(
                ids=chunk_ids,
                documents=chunks,
                metadatas=[{"source": filename, "chunk_id": i, "type": "text"} for i in range(len(chunks))]
            )
            logging.info(f"Added {len(chunks)} text chunks to ChromaDB.")

        # Add image documents to ChromaDB
        if image_docs_to_index:
            collection.add(
                ids=[d["id"] for d in image_docs_to_index],
                documents=[d["document"] for d in image_docs_to_index],
                metadatas=[d["metadata"] for d in image_docs_to_index]
            )
            logging.info(f"Added {len(image_docs_to_index)} image documents to ChromaDB.")

        return {
            "status": "success",
            "filename": filename,
            "chunks_added": len(chunks) + len(image_docs_to_index),
            "total_chunks": collection.count()
        }

    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        logging.error(f"Upload failed for {filename}:\n{tb}")
        # Clean up saved file on failure
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

@app.post("/api/clear-docs")
def clear_documents():
    try:
        # Delete files in data/
        for item in DATA_FOLDER.glob("*"):
            if item.is_file():
                item.unlink()
                
        # Delete files in extracted_images/
        for item in IMAGES_FOLDER.glob("*"):
            if item.is_file():
                item.unlink()
                
        # Also clean up any loose markdown or log files if present
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
        # Delete all chunks in Chroma collection
        existing = collection.get()
        deleted_count = 0
        if existing and existing["ids"]:
            deleted_count = len(existing["ids"])
            collection.delete(ids=existing["ids"])
            
        # Clear Query Cache
        query_cache = {}
        if CACHE_FILE.exists():
            CACHE_FILE.unlink()
            
        return {
            "status": "success", 
            "message": f"Successfully deleted {deleted_count} chunks from ChromaDB and cleared query cache."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear ChromaDB: {str(e)}")

@app.post("/api/query", response_model=QueryResponse)
def query_rag_api(req: QueryRequest):
    # Check cache first
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
            # Fallback for old string-only cache: retrieve sources/images from ChromaDB
            try:
                chroma_res = collection.query(
                    query_texts=[req.query],
                    n_results=req.n_results,
                    include=["documents", "metadatas"]
                )
                sources = []
                image_paths = []
                if chroma_res and chroma_res["documents"] and chroma_res["documents"][0]:
                    for doc, meta in zip(chroma_res["documents"][0], chroma_res["metadatas"][0]):
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
            except Exception as e:
                print(f"Error fetching sources for cached query: {e}")
                sources = []
                image_paths = []
                
            return QueryResponse(
                query=req.query,
                answer=cached_val,
                cached=True,
                sources=sources,
                image_paths=image_paths
            )
        
    llm_instance = get_llm()
    if llm_instance is None:
        raise HTTPException(status_code=400, detail="Gemini LLM is not configured. Please set GOOGLE_API_KEY.")
        
    count = collection.count()
    if count == 0:
        raise HTTPException(status_code=400, detail="No documents indexed in ChromaDB. Please upload a PDF first.")
        
    # Rate limiting sleep (from notebook)
    time_since_last_call = time.time() - last_api_call["time"]
    min_delay = 4.0
    if time_since_last_call < min_delay:
        wait_time = min_delay - time_since_last_call
        time.sleep(wait_time)
        
    try:
        # Query Chroma
        chroma_res = collection.query(
            query_texts=[req.query],
            n_results=req.n_results,
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
            
        retrieved_docs = chroma_res["documents"][0]
        retrieved_metas = chroma_res["metadatas"][0]
        
        # Build references list and collect image paths
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
            
        # Optimize prompt context length
        context = "\n\n---\n\n".join(
            f"Source: {meta.get('source', 'unknown')}, Type: {meta.get('type', 'text')}\n{doc[:req.chunk_chars]}"
            for doc, meta in zip(retrieved_docs, retrieved_metas)
        )
        
        prompt = f"Answer based on this context only.\nQuery: {req.query}\nContext: {context}\nAnswer:"
        
        # Invoke LLM
        response = llm_instance.invoke(prompt)
        answer = response.content if hasattr(response, 'content') else str(response)
        
        # Save to cache
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
        print(f"Error querying RAG: {e}")
        raise HTTPException(status_code=500, detail=f"Error executing RAG query: {str(e)}")

# Mount Static Files (placed at the end, so API routes take precedence)
static_dir = BASE_DIR / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/extracted_images", StaticFiles(directory=str(IMAGES_FOLDER)), name="images")
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
