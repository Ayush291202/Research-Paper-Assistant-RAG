# RefRAG: Research Paper Assistant (RAG)

A simple Retrieval-Augmented Generation system that answers questions about your
research papers. Built with LangChain + Chroma + Hugging Face embeddings + Ollama.

## Files
- `config.py` — all settings (paths, models, chunk size)
- `ingest.py` — reads PDFs, chunks them, builds the Chroma index
- `rag.py` — retrieves chunks and asks the LLM
- `app.py` — Streamlit UI
- `papers/` — put your PDFs here

## Setup (macOS, Apple Silicon)

1. Install Ollama and pull a model:
   ```bash
   brew install ollama
   ollama serve         # leave running in a separate terminal
   ollama pull llama3.2:1b   # or whatever model you want
   ollama list               # confirm model name
   ```
   If your model name is not `llama3.2:1b`, edit `OLLAMA_MODEL` in `config.py`.

2. Create a Python environment and install deps:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. Put PDFs in the `papers/` folder (already done for you).

## Run

1. Build the index (one-time; takes a few minutes):
   ```bash
   python ingest.py
   ```
   Re-run this whenever you add or remove papers.
2. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. Type a question and click **Ask**.

## How it works
1. `ingest.py` reads each PDF page-by-page with PyMuPDF, splits the text into
   500-character chunks (100 overlap), embeds them with `all-MiniLM-L6-v2`,
   and saves a Chroma vector store in `chroma_db/`.
2. When you ask a question, `rag.py` embeds the query, finds the top-5 most
   similar chunks, stuffs them into a prompt, and sends it to Ollama.
3. The UI shows the answer plus the source chunks (paper title + page number).

## Tweaks
- Change chunk size or top-k in `config.py`.
- Swap the LLM by changing `OLLAMA_MODEL` (any model you've pulled with Ollama).
- Swap embeddings by changing `EMBED_MODEL`.
