# ingest.py
# Read PDFs -> chunk -> embed -> save Chroma vector store (using LangChain).

import os
import shutil
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import config


def load_papers():
    """Load every PDF in PAPERS_DIR as LangChain Documents (one per page)."""
    docs = []
    pdf_files = sorted(
        f for f in os.listdir(config.PAPERS_DIR) if f.lower().endswith(".pdf")
    )
    for pdf_name in pdf_files:
        path = os.path.join(config.PAPERS_DIR, pdf_name)
        print(f"  Loading {pdf_name}...")
        loader = PyMuPDFLoader(path)
        pages = loader.load() # one Document per page, with page metadata
        pages = [p for p in pages if p.page_content.strip()] 
        title = os.path.splitext(pdf_name)[0]
        for p in pages:
            p.metadata["paper"] = title
            # PyMuPDF page indices start at 0; show them starting at 1
            p.metadata["page"] = p.metadata.get("page", 0) + 1
        docs.extend(pages)
    return docs


def build_index():
    """Process every PDF and save a Chroma vector store."""
    print("Loading PDFs...")
    docs = load_papers()
    if not docs:
        print(f"No PDFs found in {config.PAPERS_DIR}/")
        return
    print(f"Loaded {len(docs)} pages.")

    print("Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(docs)
    print(f"Total chunks: {len(chunks)}")

    print(f"Loading embedding model: {config.EMBED_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBED_MODEL)

    # Wipe any previous index so rebuilds start clean
    if os.path.exists(config.INDEX_DIR):
        shutil.rmtree(config.INDEX_DIR)

    print("Building Chroma index (this may take a few minutes)...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=config.COLLECTION_NAME,
        persist_directory=config.INDEX_DIR,
    )
    print(f"Saved index to {config.INDEX_DIR}/")
    print("Done.")


if __name__ == "__main__":
    build_index()
