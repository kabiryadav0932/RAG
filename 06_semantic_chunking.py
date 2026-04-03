"""
06_semantic_chunking.py
-----------------------
Semantic chunking: splits text where the MEANING changes, not just
at fixed character counts. Uses cosine similarity between consecutive
sentence embeddings — when similarity drops below a threshold, we start
a new chunk.

No OpenAI. Uses sentence-transformers locally.

Stack: ChromaDB + sentence-transformers (zero cost, fully local)
"""

import os
import numpy as np
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from sklearn.metrics.pairwise import cosine_similarity

# ── Config ────────────────────────────────────────────────────────────────────
DOCS_PATH            = "docs"
DB_PATH              = "db/chroma_db_semantic"   # separate DB
EMBEDDING_MODEL      = "all-MiniLM-L6-v2"
COLLECTION_NAME      = "semantic_chunks"
BREAKPOINT_THRESHOLD = 0.75   # lower = more chunks, higher = fewer bigger chunks

# ── Load Documents ────────────────────────────────────────────────────────────
def load_documents(path: str):
    loader = DirectoryLoader(path, glob="**/*.txt", loader_cls=TextLoader)
    docs   = loader.load()
    print(f"[load] Loaded {len(docs)} pages from '{path}'")
    return docs

# ── Split into Sentences ──────────────────────────────────────────────────────
def split_into_sentences(docs):
    """Naive sentence splitter — good enough for most RAG use cases."""
    sentences = []
    for doc in docs:
        raw   = doc.page_content.replace("\n", " ")
        parts = [s.strip() for s in raw.replace("! ", ".|").replace("? ", ".|").split(".|")]
        for part in parts:
            if len(part) > 20:
                sentences.append({"text": part, "metadata": doc.metadata})
    print(f"[sentences] Extracted {len(sentences)} sentences")
    return sentences

# ── Semantic Chunking ─────────────────────────────────────────────────────────
def semantic_chunk(sentences, embed_model):
    """
    Embeds each sentence, computes cosine similarity between consecutive
    sentences, and breaks into a new chunk wherever similarity < threshold.
    """
    if not sentences:
        print("[semantic_chunk] No sentences to chunk.")
        return []

    texts      = [s["text"] for s in sentences]
    embeddings = np.array(embed_model.embed_documents(texts))

    chunks                  = []
    current_chunk_sentences = [sentences[0]]

    for i in range(1, len(sentences)):
        sim = cosine_similarity([embeddings[i - 1]], [embeddings[i]])[0][0]
        if sim < BREAKPOINT_THRESHOLD:
            chunk_text = " ".join(s["text"] for s in current_chunk_sentences)
            chunks.append(Document(
                page_content=chunk_text,
                metadata=current_chunk_sentences[0]["metadata"]
            ))
            current_chunk_sentences = [sentences[i]]
        else:
            current_chunk_sentences.append(sentences[i])

    # Flush last chunk
    if current_chunk_sentences:
        chunk_text = " ".join(s["text"] for s in current_chunk_sentences)
        chunks.append(Document(
            page_content=chunk_text,
            metadata=current_chunk_sentences[0]["metadata"]
        ))

    print(f"[semantic_chunk] {len(sentences)} sentences → {len(chunks)} semantic chunks "
          f"(threshold={BREAKPOINT_THRESHOLD})")
    return chunks

# ── Embed & Store ─────────────────────────────────────────────────────────────
def build_vectorstore(chunks, embed_model):
    vectorstore = Chroma.from_documents(
        documents         = chunks,
        embedding         = embed_model,
        persist_directory = DB_PATH,
        collection_name   = COLLECTION_NAME,
    )
    print(f"[store] {len(chunks)} chunks → ChromaDB at '{DB_PATH}'")
    return vectorstore

# ── Sanity Check ──────────────────────────────────────────────────────────────
def test_retrieval(vectorstore, query: str, k: int = 3):
    print(f"\n[query] '{query}'")
    results = vectorstore.similarity_search(query, k=k)
    for i, doc in enumerate(results, 1):
        src = doc.metadata.get("source", "unknown")
        print(f"\n  Result {i} | source: {src}")
        print(f"  {doc.page_content[:300]}...")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    embed_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    docs        = load_documents(DOCS_PATH)
    sentences   = split_into_sentences(docs)
    chunks      = semantic_chunk(sentences, embed_model)
    vectorstore = build_vectorstore(chunks, embed_model)
    test_retrieval(vectorstore, "What is the main topic of the document?")
