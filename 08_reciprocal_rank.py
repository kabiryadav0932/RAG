"""
08_reciprocal_rank_fusion.py
-----------------------------
Reciprocal Rank Fusion (RRF): a simple but powerful algorithm for
combining ranked lists from multiple retrievers into one final ranking.

Formula for each document:  RRF_score = Σ 1 / (k + rank_i)
where k=60 (standard constant), rank_i = position in list i.

Here we combine results from:
  - ChromaDB vector search (semantic similarity)
  - BM25 keyword search (lexical match)

Stack: ChromaDB + sentence-transformers + rank_bm25 (zero cost, fully local)

Install dep if missing:  pip install rank_bm25
"""

import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ── Config ────────────────────────────────────────────────────────────────────
DOCS_PATH       = "docs"
DB_PATH         = "db/chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "main_chunks"   # must match 01_Ingestion.py
TOP_K           = 5
RRF_K           = 60

# FIX: chunk size matched to 01_Ingestion.py (was 512/64)
BM25_CHUNK_SIZE    = 300
BM25_CHUNK_OVERLAP = 60

# ── Load vectorstore ──────────────────────────────────────────────────────────
def load_vectorstore():
    embeddings  = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
    return vectorstore

# ── Load chunks for BM25 ─────────────────────────────────────────────────────
def load_chunks():
    loader   = DirectoryLoader(DOCS_PATH, glob="**/*.txt", loader_cls=TextLoader)
    docs     = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=BM25_CHUNK_SIZE,
        chunk_overlap=BM25_CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    print(f"[bm25] Loaded {len(chunks)} chunks for BM25 index")
    return chunks

# ── RRF Core Algorithm ────────────────────────────────────────────────────────
def reciprocal_rank_fusion(
    result_lists: list[list[Document]],
    k: int = RRF_K
) -> list[tuple[Document, float]]:
    """
    Merge multiple ranked retrieval lists into one using RRF.

    Args:
        result_lists: Each inner list is a ranked list of Documents
                      from one retriever (index 0 = most relevant).
        k:            RRF constant (default 60).

    Returns:
        List of (Document, rrf_score) tuples, sorted descending by score.
    """
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
    return [(doc_map[key], score) for key, score in ranked]

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    query = input("Enter your question: ").strip()

    # Retriever 1: ChromaDB vector search
    print("\n[retriever 1] Vector search via ChromaDB...")
    vectorstore    = load_vectorstore()
    vector_results = vectorstore.similarity_search(query, k=TOP_K)
    print(f"  → {len(vector_results)} results")

    # Retriever 2: BM25 keyword search
    print("[retriever 2] BM25 keyword search...")
    chunks         = load_chunks()
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = TOP_K
    bm25_results   = bm25_retriever.invoke(query)
    print(f"  → {len(bm25_results)} results")

    # Fuse with RRF
    print("\n[RRF] Fusing results...")
    fused = reciprocal_rank_fusion([vector_results, bm25_results])

    print(f"\n── Top {min(5, len(fused))} Fused Results (RRF) ─────────────────────")
    for rank, (doc, score) in enumerate(fused[:5], 1):
        src = doc.metadata.get("source", "unknown")
        print(f"\n  [{rank}] Score: {score:.4f} | {src}")
        print(f"  {doc.page_content[:250]}...")
