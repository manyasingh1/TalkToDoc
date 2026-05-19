import os
from dotenv import load_dotenv
import chromadb

from google import genai
from google.genai import types

from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption
)

from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.chunking import HybridChunker


# =====================
# Load API key
# =====================

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


# =====================
# ChromaDB
# =====================

db = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = db.get_or_create_collection(
    name="multimodal_docs"
)


# =====================
# Disable OCR
# =====================

pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = False

converter = DocumentConverter(
    format_options={
        InputFormat.PDF:
        PdfFormatOption(
            pipeline_options=pipeline_options
        )
    }
)


# =====================
# Chunker
# =====================

chunker = HybridChunker(
    tokenizer="sentence-transformers/all-MiniLM-L6-v2"
)


# =====================
# Read PDFs automatically
# =====================

pdf_folder = "data"

documents = []
embeddings = []
ids = []

counter = 0


for file in os.listdir(pdf_folder):

    if not file.endswith(".pdf"):
        continue

    print(f"\nProcessing: {file}")

    pdf_path = os.path.join(
        pdf_folder,
        file
    )

    result = converter.convert(
        pdf_path
    )

    doc = result.document

    chunks = list(
        chunker.chunk(doc)
    )


    for chunk in chunks:

        text = chunk.text

        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT"
            )
        )

        vector = response.embeddings[0].values

        documents.append(text)
        embeddings.append(vector)

        ids.append(
            f"{file}_{counter}"
        )

        counter += 1


# =====================
# Save everything
# =====================

collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=ids
)

print("\nIndex created successfully")
print(f"Total chunks: {len(documents)}")