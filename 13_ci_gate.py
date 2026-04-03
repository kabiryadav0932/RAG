"""
13_ci_gate.py
-------------
CI-gated evaluation pipeline for the RAG system.

What this does:
  - Runs the full pipeline (retrieval + generation)
  - Computes citation coverage and citation precision
  - Runs RAGAS metrics
  - Checks every metric against a hardcoded threshold
  - Prints a PASS / FAIL report
  - Exits with code 0 if all pass, code 1 if any fail

In a real CI setup (GitHub Actions etc.) this script is what you'd run
in your workflow — a non-zero exit code fails the pipeline automatically.

Run:
  python 13_ci_gate.py

To simulate a CI failure, lower the thresholds or break the pipeline.
"""

import re
import sys
import json
import warnings
from datetime import datetime
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

# ── CI Thresholds ─────────────────────────────────────────────────────────────
# Adjust these based on your baseline runs.
# Any metric that falls below its threshold causes a CI FAILURE (exit code 1).
THRESHOLDS = {
    "faithfulness":       0.70,   # from run 2: 0.775
    "answer_relevancy":   0.65,   # from run 2: 0.707
    "context_precision":  0.90,   # from run 2: 0.958
    "context_recall":     0.95,   # from run 2: 1.000
    "citation_coverage":  0.70,   # >= 70% of answers must contain a citation
    "citation_precision": 0.80,   # >= 80% of cited sources must be valid
}

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
    sentences = [s.strip() for s in answer.strip().split('.') if s.strip()]
    trimmed   = '. '.join(sentences[:2])
    return trimmed + '.' if trimmed and not trimmed.endswith('.') else trimmed

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

# ── Citation Metrics ──────────────────────────────────────────────────────────
def compute_citation_metrics(answers, num_sources):
    covered     = 0
    valid       = 0
    total_cited = 0

    for answer in answers:
        cited_indices = re.findall(r'\[Source\s*(\d+)\]', answer, re.IGNORECASE)
        if cited_indices:
            covered += 1
        valid       += sum(1 for idx in cited_indices if 1 <= int(idx) <= num_sources)
        total_cited += len(cited_indices)

    coverage  = covered / len(answers) if answers else 0.0
    precision = valid / total_cited    if total_cited > 0 else 0.0
    return coverage, precision

# ── Gate Check ────────────────────────────────────────────────────────────────
def run_gate(scores: dict) -> bool:
    """
    Compares every score against its threshold.
    Returns True if ALL pass, False if ANY fail.
    Prints a formatted report either way.
    """
    print("\n" + "=" * 60)
    print("  CI GATE REPORT")
    print("=" * 60)
    print(f"  {'Metric':<24} {'Score':>7}  {'Threshold':>9}  {'Status'}")
    print(f"  {'-'*24}  {'-'*7}  {'-'*9}  {'-'*6}")

    all_pass = True
    for metric, score in scores.items():
        threshold = THRESHOLDS.get(metric, 0.0)
        passed    = score >= threshold
        status    = "PASS" if passed else "FAIL"
        marker    = "  " if passed else "* "
        if not passed:
            all_pass = False
        print(f"  {marker}{metric:<22} {score:>7.4f}  {threshold:>9.4f}  {status}")

    print("=" * 60)
    if all_pass:
        print("  RESULT : ALL CHECKS PASSED — pipeline is healthy.")
    else:
        print("  RESULT : ONE OR MORE CHECKS FAILED — pipeline regression detected.")
    print("=" * 60)
    return all_pass

# ── Save Report ───────────────────────────────────────────────────────────────
def save_report(scores: dict, passed: bool):
    report = {
        "timestamp":  datetime.now().isoformat(),
        "passed":     passed,
        "scores":     scores,
        "thresholds": THRESHOLDS,
        "failures":   [
            m for m, s in scores.items()
            if s < THRESHOLDS.get(m, 0.0)
        ],
    }
    path = "ci_gate_report.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[saved] CI report → {path}")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  CI Evaluation Gate (fully local)")
    print("=" * 60)
    print(f"  Thresholds: {THRESHOLDS}")

    print("\n[init] Loading pipeline...")
    llm            = OllamaLLM(model=OLLAMA_MODEL)
    vectorstore    = load_vectorstore()
    bm25_retriever = load_bm25_retriever()
    print("[init] Pipeline ready.")

    questions, answers, contexts, ground_truths = run_pipeline(
        TEST_CASES, vectorstore, bm25_retriever, llm
    )

    # ── Citation metrics (fast, no LLM) ──────────────────────────────────────
    print("\n[citation] Computing citation metrics...")
    cov, prec = compute_citation_metrics(answers, num_sources=TOP_K)
    print(f"  citation_coverage  : {cov:.4f}")
    print(f"  citation_precision : {prec:.4f}")

    # ── RAGAS metrics ─────────────────────────────────────────────────────────
    print("\n[ragas] Building evaluation dataset...")
    eval_dataset = Dataset.from_dict({
        "user_input":         questions,
        "response":           answers,
        "retrieved_contexts": contexts,
        "reference":          ground_truths,
    })

    print("[ragas] Configuring local judge...")
    judge_llm        = ChatOllama(model=OLLAMA_MODEL)
    judge_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    ragas_llm        = LangchainLLMWrapper(judge_llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(judge_embeddings)

    ragas_metrics = [
        Faithfulness(),
        AnswerRelevancy(),
        ContextPrecision(),
        ContextRecall(),
    ]

    for metric in ragas_metrics:
        metric.llm = ragas_llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = ragas_embeddings

    print("[ragas] Running evaluation (sequential, ~20-30 min on CPU)...\n")
    results    = evaluate(
        dataset=eval_dataset,
        metrics=ragas_metrics,
        run_config=RunConfig(max_workers=1, timeout=120, max_retries=2),
    )
    results_df = results.to_pandas()

    # ── Collect all scores ────────────────────────────────────────────────────
    ragas_metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    all_scores = {
        m: float(results_df[m].mean())
        for m in ragas_metric_names
        if m in results_df.columns
    }
    all_scores["citation_coverage"]  = cov
    all_scores["citation_precision"] = prec

    # ── Run gate ──────────────────────────────────────────────────────────────
    passed = run_gate(all_scores)
    save_report(all_scores, passed)

    # ── Exit code for CI ──────────────────────────────────────────────────────
    sys.exit(0 if passed else 1)