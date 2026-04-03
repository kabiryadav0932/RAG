"""
07_multi_query_retrieval.py
---------------------------
Multi-query retrieval: instead of searching with just the user's original
query, we use Ollama to generate N alternative phrasings of the same
question, retrieve docs for each, then deduplicate and merge results.

Why: A single query can miss relevant chunks due to vocabulary mismatch.
Multiple phrasings increases recall significantly.

Stack: ChromaDB + sentence-transformers + Ollama (zero cost, fully local)
"""

import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.documents import Document

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH         = "db/chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "main_chunks"   # must match 01_Ingestion.py
OLLAMA_MODEL    = "qwen2.5:1.5b"  # FIX: was llama3.2
NUM_QUERIES     = 3
TOP_K           = 3

# ── Load vectorstore ──────────────────────────────────────────────────────────
def load_vectorstore():
    embeddings  = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
    print(f"[load] ChromaDB loaded from '{DB_PATH}' (collection: '{COLLECTION_NAME}')")
    return vectorstore

# ── Generate Alternative Queries ──────────────────────────────────────────────
def generate_alternative_queries(original_query: str, llm) -> list[str]:
    prompt = f"""You are an expert at reformulating search queries.
Given the following question, generate {NUM_QUERIES} different versions of it
that mean the same thing but use different words or perspectives.
Return ONLY the queries, one per line, no numbering, no explanations.

Original question: {original_query}

Alternative queries:"""

    response     = llm.invoke(prompt)
    alternatives = [
        line.strip()
        for line in response.strip().split("\n")
        if line.strip() and len(line.strip()) > 10
    ][:NUM_QUERIES]

    print(f"\n[multi_query] Original: '{original_query}'")
    for i, q in enumerate(alternatives, 1):
        print(f"  Alt {i}: {q}")

    return [original_query] + alternatives

# ── Retrieve & Deduplicate ────────────────────────────────────────────────────
def multi_query_retrieve(queries: list[str], vectorstore) -> list[Document]:
    """
    Run each query against the vectorstore, collect all results,
    deduplicate by page content.
    """
    seen_contents = set()
    unique_docs   = []

    for query in queries:
        results = vectorstore.similarity_search(query, k=TOP_K)
        for doc in results:
            if doc.page_content not in seen_contents:
                seen_contents.add(doc.page_content)
                unique_docs.append(doc)

    print(f"\n[retrieve] {len(queries)} queries → {len(unique_docs)} unique docs retrieved")
    return unique_docs

# ── Generate Answer ───────────────────────────────────────────────────────────
def generate_answer(question: str, docs: list[Document], llm) -> str:
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    prompt  = f"""Answer the question using only the context provided below.
If the answer is not in the context, say "I don't know based on the given documents."

Context:
{context}

Question: {question}

Answer:"""
    return llm.invoke(prompt)

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    llm         = OllamaLLM(model=OLLAMA_MODEL)
    vectorstore = load_vectorstore()

    while True:
        question = input("\nEnter your question (or 'quit'): ").strip()
        if not question:
            continue
        if question.lower() in ("quit", "q", "exit"):
            break

        queries = generate_alternative_queries(question, llm)
        docs    = multi_query_retrieve(queries, vectorstore)

        print("\n── Retrieved Chunks ──────────────────────────────────────────")
        for i, doc in enumerate(docs, 1):
            src = doc.metadata.get("source", "unknown")
            print(f"\n  [{i}] {src}\n  {doc.page_content[:200]}...")

        print("\n── Answer ────────────────────────────────────────────────────")
        answer = generate_answer(question, docs, llm)
        print(answer)
