"""
05_recursive_chunking.py
------------------------
Demonstrates RecursiveCharacterTextSplitter — the go-to chunking
strategy for most RAG pipelines. Splits on natural boundaries
(paragraphs → sentences → words) before falling back to characters.

Difference from 01_Ingestion.py (CharacterTextSplitter):
  RecursiveCharacterTextSplitter tries a priority list of separators
  before splitting arbitrarily — produces cleaner, more coherent chunks.

NOTE: Uses its own separate DB (db/chroma_db_recursive) so it
doesn't overwrite the main pipeline DB used by files 01–04 and 07–10.

Stack: ChromaDB + sentence-transformers (zero cost, fully local)
"""

import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ── Config ────────────────────────────────────────────────────────────────────
DOCS_PATH       = "docs"
DB_PATH         = "db/chroma_db_recursive"   # separate DB — won't touch main pipeline
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "recursive_chunks"

CHUNK_SIZE    = 300   # FIX: matched to main pipeline (was 512)
CHUNK_OVERLAP = 60    # FIX: matched to main pipeline (was 64)

# ── Load Documents ────────────────────────────────────────────────────────────
def load_documents(path: str):
    loader = DirectoryLoader(path, glob="**/*.txt", loader_cls=TextLoader)
    docs   = loader.load()
    print(f"[load] Loaded {len(docs)} pages from '{path}'")
    return docs

# ── Chunk ─────────────────────────────────────────────────────────────────────
def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size      = CHUNK_SIZE,
        chunk_overlap   = CHUNK_OVERLAP,
        # Priority order: paragraph → newline → space → character
        separators      = ["\n\n", "\n", " ", ""],
        length_function = len,
    )
    chunks = splitter.split_documents(docs)
    print(f"[chunk] {len(docs)} pages → {len(chunks)} chunks "
          f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks

# ── Embed & Store ─────────────────────────────────────────────────────────────
def build_vectorstore(chunks):
    embeddings  = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(
        documents         = chunks,
        embedding         = embeddings,
        persist_directory = DB_PATH,
        collection_name   = COLLECTION_NAME,
    )
    print(f"[store] {len(chunks)} chunks embedded → ChromaDB at '{DB_PATH}'")
    return vectorstore

# ── Sanity Check Query ────────────────────────────────────────────────────────
def test_retrieval(vectorstore, query: str, k: int = 3):
    print(f"\n[query] '{query}'")
    results = vectorstore.similarity_search(query, k=k)
    for i, doc in enumerate(results, 1):
        src = doc.metadata.get("source", "unknown")
        print(f"\n  Result {i} | source: {src}")
        print(f"  {doc.page_content[:300]}...")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    docs        = load_documents(DOCS_PATH)
    chunks      = chunk_documents(docs)
    vectorstore = build_vectorstore(chunks)
    test_retrieval(vectorstore, "What is the main topic of the document?")
