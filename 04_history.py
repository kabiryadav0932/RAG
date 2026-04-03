from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage, AIMessage

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

model        = OllamaLLM(model=OLLAMA_MODEL)
chat_history = []


# ── Ask Question ──────────────────────────────────────────────────────────────
def ask_question(user_question):
    print(f"\n--- You asked: {user_question} ---")

    # Step 1: Rewrite question using chat history for better retrieval
    if chat_history:
        history_text = "\n".join([
            f"User: {msg.content}" if isinstance(msg, HumanMessage) else f"Assistant: {msg.content}"
            for msg in chat_history
        ])
        rewrite_prompt = f"""Given this chat history:
{history_text}

Rewrite this new question to be standalone and searchable (no pronouns, full context).
Return only the rewritten question, nothing else.
New question: {user_question}"""

        search_question = model.invoke(rewrite_prompt).strip()
        print(f"Rewritten for search: {search_question}")
    else:
        search_question = user_question

    # Step 2: Find relevant documents
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 3,
            "score_threshold": 0.1
        }
    )
    docs = retriever.invoke(search_question)
    print(f"Found {len(docs)} relevant documents:")
    for i, doc in enumerate(docs, 1):
        print(f"  Doc {i}:\n{doc.page_content}\n")

    # Step 3: Build prompt with history + docs
    history_text = "\n".join([
        f"User: {msg.content}" if isinstance(msg, HumanMessage) else f"Assistant: {msg.content}"
        for msg in chat_history
    ])

    combined_input = f"""You are a helpful assistant that answers questions based on provided documents and conversation history.

Conversation so far:
{history_text if chat_history else "No previous conversation."}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in docs]) if docs else "No relevant documents found."}

Answer this question using only the documents above: {user_question}
If the answer isn't in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""

    # Step 4: Get answer
    answer = model.invoke(combined_input)

    # Step 5: Save to history
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))

    print(f"Answer: {answer}")
    return answer


# ── Chat Loop ─────────────────────────────────────────────────────────────────
def start_chat():
    print("Ask me questions! Type 'quit' to exit.")
    while True:
        question = input("\nYour question: ").strip()
        if not question:
            continue
        if question.lower() in ("quit", "q", "exit"):
            print("Goodbye!")
            break
        ask_question(question)


if __name__ == "__main__":
    start_chat()
