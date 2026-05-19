import os
import io
from dotenv import load_dotenv
import chromadb

from google import genai
from google.genai import types

from PIL import Image

from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption
)

from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.chunking import HybridChunker


# ==========================
# Load API key
# ==========================

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


# ==========================
# Create/Open ChromaDB
# ==========================

db = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = db.get_or_create_collection(
    name="multimodal_docs"
)


# ==========================
# Existing IDs
# ==========================

existing = collection.get()

existing_ids = set(
    existing["ids"]
)


# ==========================
# Image description function
# ==========================

def describe_image(pil_img):

    try:

        buffer = io.BytesIO()

        pil_img.save(
            buffer,
            format="PNG"
        )

        image_bytes = buffer.getvalue()

        response = client.models.generate_content(
            model="gemini-2.5-flash",

            contents=[

                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/png"
                ),

                "Describe this image in detail including diagrams, labels, objects, charts and meaning."
            ]
        )

        return response.text

    except Exception as e:

        print(
            "Image processing error:",
            e
        )

        return None


# ==========================
# Docling settings
# ==========================

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


chunker = HybridChunker(
    tokenizer="sentence-transformers/all-MiniLM-L6-v2"
)


# ==========================
# Process PDFs
# ==========================

pdf_folder = "data"


for file in os.listdir(pdf_folder):

    if not file.endswith(".pdf"):
        continue


    if any(file in x for x in existing_ids):

        print(
            f"Skipping {file}"
        )

        continue


    print(
        f"\nIndexing {file}"
    )


    pdf_path = os.path.join(
        pdf_folder,
        file
    )


    try:

        result = converter.convert(
            pdf_path
        )

        doc = result.document


        # ==========================
        # IMAGE PROCESSING
        # ==========================

        if hasattr(doc, "pictures"):

            print(
                "Checking images..."
            )

            for pic_num, picture in enumerate(doc.pictures):

                try:

                    image = picture.image.pil_image


                    description = describe_image(
                        image
                    )


                    if description:

                        response = client.models.embed_content(

                            model="gemini-embedding-001",

                            contents=description,

                            config=types.EmbedContentConfig(
                                task_type="RETRIEVAL_DOCUMENT"
                            )
                        )


                        collection.add(

                            documents=[
                                description
                            ],

                            embeddings=[
                                response.embeddings[0].values
                            ],

                            ids=[
                                f"{file}_image_{pic_num}"
                            ],

                            metadatas=[

                                {
                                    "source": file,
                                    "type": "image"
                                }

                            ]
                        )


                        print(
                            f"Image {pic_num} indexed"
                        )


                except Exception as e:

                    print(
                        "Image failed:",
                        e
                    )


        # ==========================
        # TEXT PROCESSING
        # ==========================

        chunks = list(
            chunker.chunk(doc)
        )


        for i, chunk in enumerate(chunks):

            text = chunk.text


            try:

                response = client.models.embed_content(

                    model="gemini-embedding-001",

                    contents=text,

                    config=types.EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT"
                    )
                )


                collection.add(

                    documents=[
                        text
                    ],

                    embeddings=[
                        response.embeddings[0].values
                    ],

                    ids=[
                        f"{file}_{i}"
                    ],

                    metadatas=[

                        {
                            "source": file,
                            "type": "text"
                        }

                    ]
                )


            except Exception as e:

                print(
                    "Chunk failed:",
                    e
                )


    except Exception as e:

        print(
            f"Error in {file}:",
            e
        )


print(
    "\nUpdate complete"
)