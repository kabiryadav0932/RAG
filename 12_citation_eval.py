"""
12_citation_eval.py
-------------------
Extends 11_ragas_eval.py with citation enforcement.

What this adds:
  - Forces the LLM to cite sources inline using [Source N] notation
  - Measures citation coverage  : % of answers that contain at least one citation
  - Measures citation precision : % of cited sources that actually exist
  - Still runs full RAGAS metrics for comparison against 11_ragas_eval.py baseline

Run:
  python 12_citation_eval.py
"""

import re
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from datasets import Dataset
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM, ChatOllama
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas import evaluate
from ragas.run_config import RunConfig

# ── Config ────────────────────────────────────────────────────────────────────
DOCS_PATH          = "docs"
DB_PATH            = "db/chroma_db"
EMBEDDING_MODEL    = "all-MiniLM-L6-v2"
COLLECTION_NAME    = "main_chunks"
OLLAMA_MODEL       = "qwen2.5:1.5b"
TOP_K              = 3
RRF_K              = 60
BM25_CHUNK_SIZE    = 500
BM25_CHUNK_OVERLAP = 50

# ── Test Dataset ──────────────────────────────────────────────────────────────
TEST_CASES = [
    {
        "question": "What is Kabir's favourite movie?",
        "ground_truth": "Kabir's all-time favourite movie is Veer-Zaara, a Bollywood romantic drama."
    },
    {
        "question": "Does Kabir overthink?",
        "ground_truth": "Yes, Kabir tends to overthink at times. He processes things internally since he does not rely heavily on a large social circle."
    },
    {
        "question": "What does failure mean to Kabir?",
        "ground_truth": "Kabir sees failure as feedback. It highlights weaknesses, pushes improvement, and builds resilience. He does not fear failure."
    },
    {
        "question": "What is Kabir's core philosophy in life?",
        "ground_truth": "Life is short and should be lived with intention. Kabir believes in building something meaningful rather than chasing temporary comfort."
    },
    {
        "question": "What motivates Kabir daily?",
        "ground_truth": "The desire to improve himself and not waste his potential. He does not want to live an average life."
    },
    {
        "question": "How does Kabir handle pressure?",
        "ground_truth": "Under pressure, Kabir consciously slows himself down, takes a deep breath, and works step by step instead of panicking."
    },
    {
        "question": "What are Kabir's long-term goals?",
        "ground_truth": "Kabir aims to establish himself as a skilled professional in ML and DL, work on impactful systems, and pursue a Master's degree from a top university."
    },
    {
        "question": "What kind of people does Kabir respect?",
        "ground_truth": "Kabir respects people who are genuinely kind, respectful, and authentic. Intelligence impresses him but humility and good character earn his respect."
    },
    {
        "question": "What kind of music does Kabir prefer?",
        "ground_truth": "Kabir has a strong preference for classic Bollywood music, especially from the 90s and early 2000s."
    },
    {
        "question": "How does Kabir make decisions?",
        "ground_truth": "Kabir takes time to think deeply, evaluates the situation from different angles, considers possible outcomes, and then commits fully."
    },
]

# ── Load vectorstore ──────────────────────────────────────────────────────────
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )

# ── Load chunks for BM25 ─────────────────────────────────────────────────────
def load_bm25_retriever():
    loader   = DirectoryLoader(DOCS_PATH, glob="**/*.txt", loader_cls=TextLoader)
    docs     = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=BM25_CHUNK_SIZE,
        chunk_overlap=BM25_CHUNK_OVERLAP
    )
    chunks         = splitter.split_documents(docs)
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = TOP_K
    print(f"[bm25] Indexed {len(chunks)} chunks")
    return bm25_retriever

# ── RRF ───────────────────────────────────────────────────────────────────────
def reciprocal_rank_fusion(result_lists, k=RRF_K):
    scores  = {}
    doc_map = {}
    for result_list in result_lists:
        for rank, doc in enumerate(result_list, start=1):
            key = doc.page_content.strip()
            if key not in scores:
                scores[key]  = 0.0
                doc_map[key] = doc
            scores[key] += 1.0 / (k + rank)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[key] for key, _ in ranked]

# ── Hybrid Retrieve ───────────────────────────────────────────────────────────
def hybrid_retrieve(query, vectorstore, bm25_retriever):
    vector_results = vectorstore.similarity_search(query, k=TOP_K)
    bm25_results   = bm25_retriever.invoke(query)
    fused          = reciprocal_rank_fusion([vector_results, bm25_results])
    return fused[:TOP_K]

# ── Generate Answer WITH Citations ───────────────────────────────────────────
def generate_answer(question, docs, llm):
    context = "\n\n---\n\n".join(
        f"[Source {i+1}]: {doc.page_content}"
        for i, doc in enumerate(docs)
    )
    prompt = f"""You are a precise fact extraction assistant.
Answer the question in 1-2 sentences using ONLY the context below.
You MUST cite the source inline using [Source N] notation at the end of the sentence.
Example: "Kabir prefers working late at night [Source 2]."
Do NOT add explanations or extra details. State the answer directly.
If the answer is not in the context, say: "I don't know based on the given documents."

Context:
{context}

Question: {question}

Answer (with citation):"""

    answer = llm.invoke(prompt)
    # Hard cap: trim to first 2 sentences
    sentences = [s.strip() for s in answer.strip().split('.') if s.strip()]
    trimmed   = '. '.join(sentences[:2])
    return trimmed + '.' if trimmed and not trimmed.endswith('.') else trimmed

# ── Citation Metrics ──────────────────────────────────────────────────────────
def compute_citation_metrics(answers, num_sources):
    """
    citation_coverage  : fraction of answers containing at least one [Source N]
    citation_precision : fraction of cited [Source N] tags that refer to a
                         valid source index (1 <= N <= num_sources retrieved)
    """
    covered   = 0
    valid     = 0
    total_cited = 0

    per_question = []

    for answer in answers:
        cited_indices = re.findall(r'\[Source\s*(\d+)\]', answer, re.IGNORECASE)
        has_citation  = len(cited_indices) > 0
        if has_citation:
            covered += 1

        valid_count = sum(1 for idx in cited_indices if 1 <= int(idx) <= num_sources)
        total_cited  += len(cited_indices)
        valid        += valid_count

        per_question.append({
            "answer":       answer,
            "cited":        cited_indices,
            "has_citation": has_citation,
            "valid":        valid_count,
            "total":        len(cited_indices),
        })

    coverage  = covered / len(answers) if answers else 0.0
    precision = valid / total_cited    if total_cited > 0 else 0.0

    return coverage, precision, per_question

# ── Run Pipeline ──────────────────────────────────────────────────────────────
def run_pipeline(test_cases, vectorstore, bm25_retriever, llm):
    questions     = []
    answers       = []
    contexts      = []
    ground_truths = []

    print(f"\n[eval] Running {len(test_cases)} test cases...\n")

    for i, tc in enumerate(test_cases, 1):
        question     = tc["question"]
        ground_truth = tc["ground_truth"]

        print(f"  [{i}/{len(test_cases)}] {question}")

        docs   = hybrid_retrieve(question, vectorstore, bm25_retriever)
        answer = generate_answer(question, docs, llm)

        questions.append(question)
        answers.append(answer)
        contexts.append([doc.page_content for doc in docs])
        ground_truths.append(ground_truth)

        print(f"         → {answer[:100].strip()}...")

    return questions, answers, contexts, ground_truths

# ── Print Citation Report ─────────────────────────────────────────────────────
def print_citation_report(coverage, precision, per_question, questions):
    print("\n" + "=" * 60)
    print("  CITATION ENFORCEMENT RESULTS")
    print("=" * 60)
    bar_cov  = "█" * int(coverage  * 20)
    bar_prec = "█" * int(precision * 20)
    print(f"  {'citation_coverage':<22} {coverage:.4f}  |{bar_cov:<20}|")
    print(f"  {'citation_precision':<22} {precision:.4f}  |{bar_prec:<20}|")
    print("=" * 60)

    print("\n[per-question citation breakdown]")
    for i, (pq, q) in enumerate(zip(per_question, questions)):
        status = "CITED" if pq["has_citation"] else "MISSING"
        print(f"\n  Q{i+1}: {q[:55]}...")
        print(f"    status    : {status}")
        print(f"    answer    : {pq['answer'][:90]}...")
        if pq["cited"]:
            print(f"    sources   : {pq['cited']}  (valid: {pq['valid']}/{pq['total']})")

# ── Print RAGAS Results ───────────────────────────────────────────────────────
def print_ragas_results(results_df):
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    print("\n" + "=" * 60)
    print("  RAGAS EVALUATION RESULTS")
    print("=" * 60)

    for metric in metrics:
        if metric in results_df.columns:
            score = results_df[metric].mean()
            bar   = "█" * int(score * 20)
            print(f"  {metric:<22} {score:.4f}  |{bar:<20}|")

    print("=" * 60)
    available = [m for m in metrics if m in results_df.columns]
    if available:
        overall = sum(results_df[m].mean() for m in available) / len(available)
        print(f"  {'Overall':<22} {overall:.4f}")
    print("=" * 60)

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Citation Enforcement Evaluation (fully local)")
    print("=" * 60)

    print("\n[init] Loading pipeline...")
    llm            = OllamaLLM(model=OLLAMA_MODEL)
    vectorstore    = load_vectorstore()
    bm25_retriever = load_bm25_retriever()
    print("[init] Pipeline ready.")

    questions, answers, contexts, ground_truths = run_pipeline(
        TEST_CASES, vectorstore, bm25_retriever, llm
    )

    # ── Citation metrics (no LLM needed, pure regex) ──────────────────────────
    print("\n[citation] Computing citation metrics...")
    coverage, precision, per_question = compute_citation_metrics(answers, num_sources=TOP_K)
    print_citation_report(coverage, precision, per_question, questions)

    # ── RAGAS metrics ─────────────────────────────────────────────────────────
    print("\n[ragas] Building evaluation dataset...")
    eval_dataset = Dataset.from_dict({
        "user_input":         questions,
        "response":           answers,
        "retrieved_contexts": contexts,
        "reference":          ground_truths,
    })

    print("[ragas] Configuring local judge (Ollama + HuggingFace embeddings)...")
    judge_llm        = ChatOllama(model=OLLAMA_MODEL)
    judge_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    ragas_llm        = LangchainLLMWrapper(judge_llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(judge_embeddings)

    metrics = [
        Faithfulness(),
        AnswerRelevancy(),
        ContextPrecision(),
        ContextRecall(),
    ]

    for metric in metrics:
        metric.llm = ragas_llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = ragas_embeddings

    print("[ragas] Running evaluation (sequential, ~20-30 min on CPU)...\n")
    results = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        run_config=RunConfig(
            max_workers=1,
            timeout=120,
            max_retries=2,
        )
    )

    results_df = results.to_pandas()
    results_df.to_json("citation_ragas_results.json", orient="records", indent=2)
    print("\n[saved] Raw results → citation_ragas_results.json")
    print_ragas_results(results_df)