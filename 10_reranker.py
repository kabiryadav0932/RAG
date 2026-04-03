"""
10_reranker.py
--------------
Cross-encoder reranking: after initial retrieval (vector/hybrid), pass
each (query, chunk) pair through a cross-encoder model that jointly
encodes both and produces a much more accurate relevance score.

Two-stage pipeline:
  Stage 1 — Fast bi-encoder retrieval (ChromaDB) → get top 20 candidates
  Stage 2 — Accurate cross-encoder reranking     → reorder to top K

Why: Bi-encoders (sentence-transformers) encode query & doc separately.
Cross-encoders attend over both together — much more accurate but slower.
Using them only in stage 2 keeps latency acceptable.

Model used: cross-encoder/ms-marco-MiniLM-L-2-v2
  - FIX: switched from L-6 to L-2 — same accuracy class, ~3x faster on CPU
  - ~40MB download vs ~80MB for L-6

Stack: ChromaDB + sentence-transformers + Ollama (zero cost, fully local)

Install dep if missing:  pip install sentence-transformers
"""

import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH             = "db/chroma_db"
EMBEDDING_MODEL     = "all-MiniLM-L6-v2"
COLLECTION_NAME     = "main_chunks"   # must match 01_Ingestion.py
OLLAMA_MODEL        = "qwen2.5:1.5b"  # FIX: was llama3.2
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-2-v2"  # FIX: L-2 is faster on CPU

CANDIDATE_K = 20   # how many to fetch from ChromaDB before reranking
FINAL_K     = 4    # how many to keep after reranking

# ── Load vectorstore ──────────────────────────────────────────────────────────
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )

# ── Stage 1: Candidate Retrieval ──────────────────────────────────────────────
def retrieve_candidates(query: str, vectorstore, k: int = CANDIDATE_K) -> list[Document]:
    results = vectorstore.similarity_search(query, k=k)
    print(f"[stage 1] Retrieved {len(results)} candidates from ChromaDB")
    return results

# ── Stage 2: Cross-Encoder Reranking ─────────────────────────────────────────
def rerank(query: str, docs: list[Document], cross_encoder, top_k: int = FINAL_K):
    """
    Score each (query, doc) pair with the cross-encoder.
    Returns top_k docs sorted by cross-encoder score descending.
    """
    pairs  = [(query, doc.page_content) for doc in docs]
    scores = cross_encoder.predict(pairs)

    scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

    print(f"[stage 2] Reranked {len(docs)} candidates → keeping top {top_k}")
    for i, (doc, score) in enumerate(scored_docs[:top_k], 1):
        src = doc.metadata.get("source", "?")
        print(f"  [{i}] score={score:.4f} | {src} | {doc.page_content[:80]}...")

    return [doc for doc, _ in scored_docs[:top_k]]

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
Cite the source (e.g. [Source 1]).

Context:
{context}

Question: {question}

Answer:"""
    return llm.invoke(prompt)

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[init] Loading models...")
    llm           = OllamaLLM(model=OLLAMA_MODEL)
    vectorstore   = load_vectorstore()
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    print("[init] Ready. Cross-encoder loaded.\n")

    while True:
        question = input("Question (or 'quit'): ").strip()
        if not question:
            continue
        if question.lower() in ("quit", "q", "exit"):
            break

        candidates = retrieve_candidates(question, vectorstore)
        final_docs = rerank(question, candidates, cross_encoder)

        print("\n── Final Docs (after reranking) ──────────────────────────")
        for i, doc in enumerate(final_docs, 1):
            src = doc.metadata.get("source", "unknown")
            print(f"  [{i}] {src}\n  {doc.page_content[:200]}...\n")

        print("── Answer ────────────────────────────────────────────────")
        answer = generate_answer(question, final_docs, llm)
        print(answer, "\n")
