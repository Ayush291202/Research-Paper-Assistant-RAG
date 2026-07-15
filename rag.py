# rag.py
# Load Chroma index, retrieve top-k chunks, ask Ollama via LangChain.

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

import config


_vectorstore = None
_llm = None


def load_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        embeddings = HuggingFaceEmbeddings(model_name=config.EMBED_MODEL)
        _vectorstore = Chroma(
            collection_name=config.COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=config.INDEX_DIR,
        )
    return _vectorstore


def load_llm():
    global _llm
    if _llm is None:
        _llm = OllamaLLM(model=config.OLLAMA_MODEL)
    return _llm


PROMPT = PromptTemplate.from_template(
    """Answer the question using ONLY the context below.
Cite each claim as [Paper Title, Page X] using the sources provided.
If the answer is not in the context, say so.

Context:
{context}

Question: {question}

Answer:"""
)


def retrieve(query, k=config.TOP_K):
    """Return the top-k LangChain Documents most similar to query."""
    vs = load_vectorstore()
    return vs.similarity_search(query, k=k)


def format_context(docs):
    parts = []
    for i, d in enumerate(docs, start=1):
        paper = d.metadata.get("paper", "Unknown")
        page = d.metadata.get("page", 0)
        parts.append(f"[Source {i} — {paper}, page {page}]\n{d.page_content}")
    return "\n\n".join(parts)


def answer_question(query, k=config.TOP_K):
    docs = retrieve(query, k=k)
    context = format_context(docs)
    prompt = PROMPT.format(context=context, question=query)
    llm = load_llm()
    answer = llm.invoke(prompt).strip()
    return answer, docs


if __name__ == "__main__":
    q = input("Question: ")
    a, srcs = answer_question(q)
    print("\n--- Answer ---")
    print(a)
    print("\n--- Sources ---")
    for s in srcs:
        print(f"  {s.metadata.get('paper')} (page {s.metadata.get('page')})")
