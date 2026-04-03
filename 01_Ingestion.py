import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ── Config ────────────────────────────────────────────────────────────────────
DOCS_PATH       = "docs"
DB_PATH         = "db/chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "main_chunks"   # shared across all pipeline files
CHUNK_SIZE      = 500             # increased from 300 — keeps full Q&A pairs together
CHUNK_OVERLAP   = 50              # reduced from 60 — less redundancy between chunks


def get_embedding_model():
    """Return the embedding model used across the pipeline."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


# ── Load Documents ────────────────────────────────────────────────────────────
def load_documents(docs_path=DOCS_PATH):
    """
    Load all .txt files from the docs directory.
    Each file becomes a Document object with content + metadata (source path).
    """
    if not os.path.exists(docs_path):
        raise FileNotFoundError(
            f"Directory '{docs_path}' not found. Create it and add your .txt files."
        )

    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader
    )

    documents = loader.load()

    if not documents:
        raise FileNotFoundError(
            f"No .txt files found in '{docs_path}'. Please add your documents."
        )

    print(f"Loaded {len(documents)} document(s):\n")
    for doc in documents:
        print(f"  • {doc.metadata['source']} ({len(doc.page_content)} characters)")

    return documents


# ── Split Documents ───────────────────────────────────────────────────────────
def split_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """
    Split documents into chunks using RecursiveCharacterTextSplitter.

    Why recursive over character splitter?
      CharacterTextSplitter splits blindly on a single separator (\\n).
      This breaks Q&A pairs apart — question in one chunk, answer in another.

      RecursiveCharacterTextSplitter tries separators in priority order:
        1. \\n\\n  — blank line between Q&A pairs (preferred split point)
        2. \\n##   — section headers (## IDENTITY, ## VALUES, etc.)
        3. \\n     — fallback: single newline
        4. space   — last resort
        5. ""      — character level (never reaches this for normal text)

    Result: each chunk contains a complete Q&A pair with its question and
    full answer together — much better retrieval precision.

    chunk_size=500: your Q&A pairs are ~200-400 chars, so 500 keeps them whole.
    chunk_overlap=50: small overlap to preserve context at boundaries.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n## ", "\n", " ", ""],
        length_function=len,
    )

    chunks = text_splitter.split_documents(documents)

    print(f"\nSplit into {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})\n")

    for i, chunk in enumerate(chunks[:3]):
        print(f"  Chunk {i+1} | {chunk.metadata['source']} | {len(chunk.page_content)} chars")
        print(f"  Preview: {chunk.page_content[:120].strip()}...")
        print()

    if len(chunks) > 3:
        print(f"  ... and {len(chunks) - 3} more chunks\n")

    return chunks


# ── Create Vector Store ───────────────────────────────────────────────────────
def create_vector_store(chunks, db_path=DB_PATH):
    """
    Embed each chunk and store it in ChromaDB.
    ChromaDB persists the vectors to disk, so this only needs to run once.
    """
    print("Embedding chunks and storing in ChromaDB...")
    print("(This may take a minute on first run — model is running locally)\n")

    embeddings = get_embedding_model()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_path,
        collection_name=COLLECTION_NAME,
        collection_metadata={"hnsw:space": "cosine"}
    )

    print(f"Vector store saved to '{db_path}'")
    print(f"Collection: '{COLLECTION_NAME}'")
    print(f"Total chunks stored: {vectorstore._collection.count()}\n")

    return vectorstore


# ── Load Existing Vector Store ────────────────────────────────────────────────
def load_existing_vector_store(db_path=DB_PATH):
    """
    Load a previously created vector store from disk.
    Must use the same embedding model and collection name used during creation.
    """
    print(f"Vector store found at '{db_path}'. Loading existing store...\n")

    embeddings = get_embedding_model()

    vectorstore = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        collection_metadata={"hnsw:space": "cosine"}
    )

    count = vectorstore._collection.count()
    print(f"Loaded vector store with {count} chunks ready for retrieval.\n")

    return vectorstore


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 45)
    print("      RAG Document Ingestion Pipeline")
    print("=" * 45 + "\n")

    if os.path.exists(DB_PATH):
        return load_existing_vector_store()

    print("No existing vector store found. Starting fresh ingestion...\n")

    documents   = load_documents()
    chunks      = split_documents(documents)
    vectorstore = create_vector_store(chunks)

    print("Ingestion complete. Your documents are ready for RAG queries.")
    return vectorstore


if __name__ == "__main__":
    main()