from flask import Flask, request, jsonify
import pickle
import faiss
from embedder import get_embeddings

with open("bm25_index/bm25.pkl", "rb") as f:
    bm25 = pickle.load(f)
with open("bm25_index/docs.pkl", "rb") as f:
    bm25_docs = pickle.load(f)

faiss_index = faiss.read_index("faiss_index/faiss.index")
with open("faiss_index/docs.pkl", "rb") as f:
    faiss_docs = pickle.load(f)

app = Flask(__name__)

@app.route("/search", methods=["POST"])
def search():
    query = request.json.get("query", "")
    if not query:
        return jsonify({"error": "No query"}), 400

    tokenized = query.split()
    bm25_scores = bm25.get_scores(tokenized)
    bm25_top = sorted(zip(bm25_docs, bm25_scores), key=lambda x: x[1], reverse=True)[:3]

    query_vec = get_embeddings([query])
    D, I = faiss_index.search(query_vec, 3)
    faiss_top = [(faiss_docs[i], float(D[0][idx])) for idx, i in enumerate(I[0])]

    return jsonify({
        "bm25_results": [doc for doc, score in bm25_top],
        "faiss_results": [doc for doc, score in faiss_top]
    })

if __name__ == "__main__":
    app.run(debug=True)
