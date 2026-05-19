import os
from dotenv import load_dotenv
import chromadb

from google import genai
from google.genai import types


load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

db = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = db.get_collection(
    name="multimodal_docs"
)


while True:

    question = input("\nAsk: ")

    if question.lower() == "exit":
        break


    query_embedding = client.models.embed_content(
        model="gemini-embedding-001",
        contents=question,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY"
        )
    )

    vector = query_embedding.embeddings[0].values


    results = collection.query(
        query_embeddings=[vector],
        n_results=5
    )


    context = "\n".join(
        results["documents"][0]
    )

    print("\nRetrieved chunks:")
    print("="*50)
    print(context[:1000])   # first 1000 chars only
    print("="*50)


    prompt = f"""
Use only the context below.

Context:
{context}

Question:
{question}
"""


    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )


    print("\nAnswer:")
    print(response.text)