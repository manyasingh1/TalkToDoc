import os
import ollama
import chromadb
from google.cloud import storage

# ==========================
# Config — UPDATE THIS
# ==========================

GCS_BUCKET = "ragproj-chromadb"   # ← same bucket as ingest.py
LOCAL_DB   = "./chroma_db"

# ==========================
# Sync chroma_db from GCS
# ==========================

def sync_from_gcs():
    print("Syncing chroma_db from Cloud Storage...")
    gcs    = storage.Client()
    bucket = gcs.bucket(GCS_BUCKET)
    blobs  = bucket.list_blobs(prefix="chroma_db/")
    for blob in blobs:
        local_path = blob.name
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)
    print("Sync complete.\n")

sync_from_gcs()

# ==========================
# ChromaDB
# ==========================

db = chromadb.PersistentClient(path=LOCAL_DB)
collection = db.get_collection(name="multimodal_docs")
print("Chunks in DB:", collection.count())
# Peek at what's stored
peek = collection.peek(5)
print("\nSample stored chunks:")
for doc, meta in zip(peek["documents"], peek["metadatas"]):
    print(f"  [{meta['type']}] {doc[:100]}")
print()

# ==========================
# Embedding
# ==========================

def embed_text(text):
    response = ollama.embeddings(
        model="nomic-embed-text",
        prompt=text
    )
    return response["embedding"]

# ==========================
# Chat Loop
# ==========================

while True:
    question = input("\nAsk: ")
    if question.lower() == "exit":
        break

    vector = embed_text(question)
    # DEBUG
    print("Vector length:", len(vector))
    print("First 5 values:", vector[:5])

    raw = collection.query(
        query_embeddings=[vector],
        n_results=3,
        include=["documents", "metadatas", "distances"]
    )
    print("Raw distances:", raw["distances"])
    print("Raw docs:", [d[:50] for d in raw["documents"][0]])

    
    results = collection.query(
        query_embeddings=[vector],
        n_results=8,
        include=["documents", "metadatas", "distances"]
    )

    if not results["documents"][0]:
        print("No relevant context found.")
        continue

    # --- Filter by relevance ---
    MAX_DISTANCE = 500

    pairs = [
        (doc, meta, dist)
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )
        if dist <= MAX_DISTANCE
    ]

    if not pairs:
        print("No sufficiently relevant context found.")
        continue

    # --- Build context ---
    context = ""
    for doc, meta, dist in pairs:
        context += f"""
SOURCE: {meta['source']}
TYPE: {meta['type']}
PAGE: {meta.get('page', 'N/A')}
DISTANCE: {round(dist, 1)}
CONTENT:
{doc[:2000]}
--------------------------------
"""

    print("\nRetrieved chunks:")
    print("=" * 50)
    print(context[:3000])
    print("=" * 50)

    # --- Generate answer ---
    prompt = f"""You are a strict document Q&A assistant.

RULES:
1. Answer ONLY using the retrieved context below
2. If something is clearly stated in context, state it confidently
3. If the question contains wrong information, correct it using the context
4. If the answer is not in context at all, say: "This is not in the provided documents"
5. Never use outside knowledge
6. Keep answers concise and direct

Retrieved Context:
{context}

Question:
{question}

Answer clearly and accurately."""

    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}]
    )

    print("\nAnswer:")
    print(response["message"]["content"])