import os
import io
import base64
import ollama
import chromadb
from google.cloud import storage
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.chunking import HybridChunker

# ==========================
# Config — UPDATE THIS
# ==========================

GCS_BUCKET  = "ragproj-chromadb"   # ← your actual GCS bucket name
LOCAL_DB    = "./chroma_db"
DATA_FOLDER = "./data"

# ==========================
# ChromaDB (local on VM)
# ==========================

db = chromadb.PersistentClient(path=LOCAL_DB)
collection = db.get_or_create_collection(name="multimodal_docs")
existing_ids = set(collection.get()["ids"])

# ==========================
# Image Filter
# ==========================

def is_useful_image(pil_img):
    w, h = pil_img.size
    if w < 100 or h < 100:
        return False
    aspect = w / h
    if aspect > 10 or aspect < 0.1:
        return False
    if w * h < 20000:
        return False
    return True

# ==========================
# Image Description (LLaVA)
# ==========================

def describe_image(pil_img):
    try:
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        response = ollama.chat(
            model="llava",
            messages=[{
                "role": "user",
                "content": """Analyze this image from a PDF. Return:
1. Main topic
2. Visible text and labels
3. Objects or diagrams present
4. Graph or chart data if applicable
5. Technical meaning
6. Summary for semantic retrieval
Be specific and structured.""",
                "images": [image_b64]
            }]
        )
        return response["message"]["content"]
    except Exception as e:
        print("Image error:", e)
        return None

# ==========================
# Embedding (nomic-embed-text)
# ==========================

def embed_text(text):
    response = ollama.embeddings(
        model="nomic-embed-text",
        prompt=text
    )
    return response["embedding"]

# ==========================
# Docling Setup
# ==========================

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

chunker = HybridChunker(
    tokenizer="sentence-transformers/all-MiniLM-L6-v2"
)

# ==========================
# Process PDFs
# ==========================

for file in os.listdir(DATA_FOLDER):
    if not file.endswith(".pdf"):
        continue
    if any(file in x for x in existing_ids):
        print(f"Skipping {file} (already indexed)")
        continue

    print(f"\nIndexing: {file}")

    try:
        result = converter.convert(os.path.join(DATA_FOLDER, file))
        doc = result.document

        # --- Images ---
        if hasattr(doc, "pictures"):
            for pic_num, picture in enumerate(doc.pictures):
                try:
                    img_id = f"{file}_image_{pic_num}"
                    if img_id in existing_ids:
                        continue
                    pil_img = picture.image.pil_image
                    if pil_img is None:
                        continue
                    if not is_useful_image(pil_img):
                        print(f"  Skipped image {pic_num} (decorative)")
                        continue

                    os.makedirs("extracted_images", exist_ok=True)
                    img_path = f"extracted_images/{file}_{pic_num}.png"
                    pil_img.save(img_path)

                    caption = ""
                    if hasattr(picture, "captions") and picture.captions:
                        caption = " ".join(
                            c.text for c in picture.captions if hasattr(c, "text")
                        )

                    print(f"  Describing image {pic_num}...")
                    description = describe_image(pil_img)
                    if not description:
                        continue

                    page_no = -1
                    if hasattr(picture, "prov") and picture.prov:
                        page_no = picture.prov[0].page_no

                    final_text = f"IMAGE DESCRIPTION:\n{description}\n\nCAPTION:\n{caption}"
                    embedding = embed_text(final_text)

                    collection.add(
                        documents=[final_text],
                        embeddings=[embedding],
                        ids=[img_id],
                        metadatas=[{
                            "source": file,
                            "type": "image",
                            "page": page_no,
                            "image_path": img_path
                        }]
                    )
                    print(f"  Image {pic_num} indexed (page {page_no})")

                except Exception as e:
                    print(f"  Image {pic_num} failed:", e)

        # --- Text Chunks ---
        for i, chunk in enumerate(chunker.chunk(doc)):
            chunk_id = f"{file}_{i}"
            if chunk_id in existing_ids:
                continue
            text = chunk.text.strip()
            if len(text) < 20:
                continue
            try:
                embedding = embed_text(text)
                collection.add(
                    documents=[text],
                    embeddings=[embedding],
                    ids=[chunk_id],
                    metadatas=[{
                        "source": file,
                        "type": "text",
                        "chunk": i,
                        "page": getattr(chunk.meta, "page_no", -1)
                    }]
                )
            except Exception as e:
                print(f"  Chunk {i} failed:", e)

        print(f"Done: {file}")

    except Exception as e:
        print(f"Error processing {file}:", e)

print("\nIndexing complete.")

# ==========================
# Sync chroma_db → GCS
# ==========================

print("\nUploading chroma_db to Cloud Storage...")

gcs = storage.Client()
bucket = gcs.bucket(GCS_BUCKET)

for root, dirs, files in os.walk(LOCAL_DB):
    for fname in files:
        local_path = os.path.join(root, fname)
        gcs_path   = local_path.replace("\\", "/").lstrip("./")
        blob       = bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)

print(f"Synced to gs://{GCS_BUCKET}/")
print("Done! Stop your VM now to save credits.")