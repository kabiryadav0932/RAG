from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH         = "db/chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "main_chunks"   # must match 01_Ingestion.py

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

# ── Query ─────────────────────────────────────────────────────────────────────
query     = input("Enter your question: ").strip()   # FIX: was hardcoded
retriever = db.as_retriever(search_kwargs={"k": 3})
relevant_docs = retriever.invoke(query)

print(f"\nUser Query: {query}")
print(f"Chunks retrieved: {len(relevant_docs)}\n")
print("--- Context ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
