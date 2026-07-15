# app.py
# Streamlit UI for the Research Paper Assistant.

import os
import streamlit as st

import config
import rag


st.set_page_config(page_title="Research Paper Assistant", layout="wide")
st.title("Research Paper Assistant")
st.caption("Ask questions about your uploaded research papers (Ollama)")


# Sidebar: list papers
with st.sidebar:
    st.header("Papers")
    if os.path.isdir(config.PAPERS_DIR):
        pdfs = sorted(
            f for f in os.listdir(config.PAPERS_DIR) if f.lower().endswith(".pdf")
        )
        st.write(f"{len(pdfs)} PDF(s) found in `{config.PAPERS_DIR}/`")
        for p in pdfs:
            st.write(f"- {p}")
    else:
        st.warning(f"Folder `{config.PAPERS_DIR}/` not found.")

    st.divider()
    st.caption(
        "To rebuild the index after adding new papers, run:\n\n"
        "`python ingest.py`"
    )


# Main area: ask a question
st.header("Ask a question")

index_ready = os.path.exists(os.path.join(config.INDEX_DIR, "chroma.sqlite3"))

if not index_ready:
    st.error(
        "No index found. Run `python ingest.py` in your terminal to build it, "
        "then refresh this page."
    )
else:
    query = st.text_input("Your question", placeholder="e.g. What is self-attention?")
    k = st.slider("Number of chunks to retrieve (top-k)", 1, 10, config.TOP_K)

    if st.button("Ask") and query.strip():
        with st.spinner("Thinking..."):
            answer, sources = rag.answer_question(query, k=k)

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Sources")
        for i, s in enumerate(sources, start=1):
            paper = s.metadata.get("paper", "Unknown")
            page = s.metadata.get("page", "?")
            with st.expander(f"[{i}] {paper} — page {page}"):
                st.write(s.page_content)
