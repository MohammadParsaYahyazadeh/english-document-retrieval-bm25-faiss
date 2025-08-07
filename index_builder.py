import os
import pickle
import numpy as np
from rank_bm25 import BM25Okapi
import faiss
from embedder import get_embeddings

os.makedirs("bm25_index", exist_ok=True)
os.makedirs("faiss_index", exist_ok=True)

with open("data/sample.txt", "r", encoding="utf-8") as f:
    docs = [line.strip() for line in f if line.strip()]

tokenized_corpus = [doc.split() for doc in docs]
bm25 = BM25Okapi(tokenized_corpus)

with open("bm25_index/bm25.pkl", "wb") as f:
    pickle.dump(bm25, f)
with open("bm25_index/docs.pkl", "wb") as f:
    pickle.dump(docs, f)

embeddings = get_embeddings(docs)
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

faiss.write_index(index, "faiss_index/faiss.index")
with open("faiss_index/docs.pkl", "wb") as f:
    pickle.dump(docs, f)

print("✅ Indexes saved.")
