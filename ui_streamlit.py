import streamlit as st
import requests

st.set_page_config(page_title="English Document Search", layout="centered")
st.title("🔍 English Document Retrieval System")

query = st.text_input("Enter your query:")

if st.button("Search"):
    if not query:
        st.warning("Please enter a query.")
    else:
        with st.spinner("Searching..."):
            try:
                res = requests.post("http://localhost:5000/search", json={"query": query})
                if res.status_code == 200:
                    results = res.json()
                    st.subheader("📌 FAISS Results:")
                    for r in results["faiss_results"]:
                        st.write(f"🟢 {r}")

                    st.subheader("📌 BM25 Results:")
                    for r in results["bm25_results"]:
                        st.write(f"🔵 {r}")
                else:
                    st.error("API error.")
            except Exception as e:
                st.error(f"Connection failed: {e}")
