from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH         = "db/chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "main_chunks"   # must match 01_Ingestion.py
OLLAMA_MODEL    = "qwen2.5:1.5b"  # FIX: was llama3.2

# ── Load vectorstore ──────────────────────────────────────────────────────────
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"}
)

db = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embedding_model,
    collection_name=COLLECTION_NAME,
)

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 3,
        "score_threshold": 0.1
    }
)

model = OllamaLLM(model=OLLAMA_MODEL)

# ── Query ─────────────────────────────────────────────────────────────────────
query = input("Enter your question: ").strip()

relevant_docs = retriever.invoke(query)
print(f"\nChunks retrieved: {len(relevant_docs)}")

print(f"User Query: {query}")
print("--- Context ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")

if not relevant_docs:
    print("No relevant documents found. Try lowering the score_threshold.")
else:
    combined_input = f"""Based on the following documents, answer this question: {query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Provide a clear answer using only the information from these documents.
If you can't find the answer, say "I don't have enough information to answer that question based on the provided documents."
"""
    result = model.invoke(combined_input)
    print("\n--- Generated Response ---")
    print(result)
