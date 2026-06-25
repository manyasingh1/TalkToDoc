# Document Parsing and Cross Conversation Project

## Introduction
In todays day and age documents have become an integral part of our professional and academic life, to some extent even personal! From important books/ material to offical reports everything have majorly shifted to online pdf documents. 

These documents are generally in pdf format. They are portable, easy to access and use. But they can be bulky. For example if you are going through a book on Machine Learning, or going through an annual report for your company--they all require undivided long term attention to be understood and read properly. Even if you're accessing something for a specific topic/ field, it would still take time to find the correct context and material in the document to properly understand it. Apart from this the material may not be easy to understand and may require multiple reads to get it. 

# Docling Hybrid RAG System

A high-performance Retrieval-Augmented Generation (RAG) backend built with **FastAPI**. This system handles complex PDF documents (including embedded tables and pictures) using **Docling**, performs hybrid retrieval combining **Semantic Search** (ChromaDB) and **Keyword Search** (BM25), and generates answers using **Google Gemini 2.5 Flash**.

---

## 🚀 Key Features

* **Advanced Document Parsing:** Uses `docling` for precise PDF text layout formatting and structured extraction of embedded images.
* **True Asynchronous Background Processing:** File processing happens entirely inside non-blocking background worker threads. FastAPI remains responsive even under heavy parsing loads.
* **Adaptive Memory-Safe Engine:** Automatically scales down picture rendering resolution or falls back to text-only mode depending on file size to prevent Out-Of-Memory (OOM) crashes.
* **Hybrid Retrieval Pipeline:** Blends Dense Semantic Embeddings (ChromaDB) with Sparse Token-matching (BM25) via **Reciprocal Rank Fusion (RRF)**.
* **Atomic Query Caching:** Safely writes query responses to local JSON cache using `threading.Lock` and atomic file system replacement (`os.replace`) to eliminate race conditions.

---

## 🛠️ Architecture Overview

The backend processing flow is split safely between web-server interactions and heavy file parsing tasks:

