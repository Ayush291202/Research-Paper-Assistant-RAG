# config.py
# All settings in one place. Change values here.

import os

# Where the PDFs live
PAPERS_DIR = "papers"

# Where Chroma persists its vector store on disk
INDEX_DIR = "chroma_db"
COLLECTION_NAME = "papers"

# Chunking
CHUNK_SIZE = 500       # characters per chunk
CHUNK_OVERLAP = 100    # overlap between chunks

# Embedding model (downloads on first run, ~80MB)
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Ollama settings
# NOTE: change this to whichever model you have pulled. Run `ollama list` to check.
# Common options: "llama3.2", "gemma3", "gemma2", "llama3", "mistral"
OLLAMA_MODEL = "llama3.2:1b"

# Retrieval
TOP_K = 5  # number of chunks to retrieve per query
