"""
09_hybrid_search.py
--------------------
Full hybrid search pipeline:
  BM25 (lexical) + ChromaDB vector search (semantic) → RRF fusion → Ollama answer

Why hybrid?
  - Vector search excels at semantic/conceptual queries
  - BM25 excels at exact keyword/entity matches (names, project titles, etc.)
  - Combining both catches what either alone misses

Stack: ChromaDB + sentence-transformers + rank_bm25 + Ollama (zero cost, fully local)

Install dep if missing:  pip install rank_bm25
"""

import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ── Config ────────────────────────────────────────────────────────────────────
DOCS_PATH       = "docs"
DB_PATH         = "db/chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "main_chunks"   # must match 01_Ingestion.py
OLLAMA_MODEL    = "qwen2.5:1.5b"  # FIX: was llama3.2
TOP_K           = 3
RRF_K           = 60

# FIX: chunk size matched to 01_Ingestion.py (was 512/64)
BM25_CHUNK_SIZE    = 500
BM25_CHUNK_OVERLAP = 50

# ── Load vectorstore ──────────────────────────────────────────────────────────
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )

# ── Load chunks for BM25 ─────────────────────────────────────────────────────
def load_chunks():
    """
    Reload raw docs and rechunk them for BM25.
    BM25 works over in-memory text — it doesn't use ChromaDB.
    Chunk size is matched to ingestion so BM25 and vector chunks are comparable.
    """
    loader   = DirectoryLoader(DOCS_PATH, glob="**/*.txt", loader_cls=TextLoader)
    docs     = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=BM25_CHUNK_SIZE,
        chunk_overlap=BM25_CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    print(f"[bm25] Indexed {len(chunks)} chunks")
    return chunks

# ── RRF ───────────────────────────────────────────────────────────────────────
def reciprocal_rank_fusion(
    result_lists: list[list[Document]],
    k: int = RRF_K
) -> list[Document]:
    scores:  dict[str, float]    = {}
    doc_map: dict[str, Document] = {}

    for result_list in result_lists:
        for rank, doc in enumerate(result_list, start=1):
            key = doc.page_content
            if key not in scores:
                scores[key]  = 0.0
                doc_map[key] = doc
            scores[key] += 1.0 / (k + rank)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[key] for key, _ in ranked]

# ── Hybrid Retrieve ───────────────────────────────────────────────────────────
def hybrid_retrieve(query: str, vectorstore, bm25_retriever) -> list[Document]:
    vector_results = vectorstore.similarity_search(query, k=TOP_K)
    bm25_results   = bm25_retriever.invoke(query)

    print(f"  [vector] {len(vector_results)} results")
    print(f"  [bm25]   {len(bm25_results)} results")

    fused = reciprocal_rank_fusion([vector_results, bm25_results])
    print(f"  [fused]  {len(fused)} unique docs after RRF → keeping top {TOP_K}")
    return fused[:TOP_K]

# ── Generate Answer ───────────────────────────────────────────────────────────
def generate_answer(question: str, docs: list[Document], llm) -> str:
    context = "\n\n---\n\n".join(
        f"[Source {i+1}]: {doc.page_content}"
        for i, doc in enumerate(docs)
    )
    prompt = f"""You are a fact extraction assistant. The context below is a personal profile.
Your job is to find and state the answer directly from the context.
If the answer exists anywhere in the context, you MUST state it. Do not say you don't know.
Only say "I don't know based on the given documents." if the answer is genuinely absent from ALL sources.
Cite the source number (e.g. [Source 1]).

Context:
{context}

Question: {question}

Answer:"""
    return llm.invoke(prompt)

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[init] Loading models and indexes...")
    llm            = OllamaLLM(model=OLLAMA_MODEL)
    vectorstore    = load_vectorstore()
    chunks         = load_chunks()
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = TOP_K
    print("[init] Ready.\n")

    while True:
        question = input("Question (or 'quit'): ").strip()
        if not question:
            continue
        if question.lower() in ("quit", "q", "exit"):
            break

        print("\n[hybrid retrieve]")
        docs = hybrid_retrieve(question, vectorstore, bm25_retriever)

        print("\n── Retrieved Chunks ──────────────────────────────────────")
        for i, doc in enumerate(docs, 1):
            src = doc.metadata.get("source", "unknown")
            print(f"  [{i}] {src}\n  {doc.page_content[:200]}...\n")

        print("── Answer ────────────────────────────────────────────────")
        answer = generate_answer(question, docs, llm)
        print(answer, "\n")
