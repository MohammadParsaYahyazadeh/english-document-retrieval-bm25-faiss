# 🔍 English Document Retrieval System (BM25 + FAISS)

A hybrid information retrieval system that uses **BM25** for lexical matching and **FAISS** for dense semantic search.  
Built with Flask API + Streamlit UI, and supports fast and accurate document matching in English.

---

## ✨ Features

- 🔤 English document search
- 🔎 BM25 ranking (lexical similarity)
- 🧠 FAISS dense retrieval using sentence-transformers
- ⚡ Fast hybrid matching: lexical + semantic
- 💡 Simple API (`/search`)
- 🌐 Streamlit UI for interactive search

---

## 🧠 Models & Algorithms

- `BM25` via `rank_bm25` (token-based)
- `FAISS` (dense vector similarity)
- Sentence embeddings via:
  - `paraphrase-multilingual-MiniLM-L12-v2` from SentenceTransformers

---

## 📁 Project Structure

