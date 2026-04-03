# Another Me — Local RAG Pipeline

A fully local, zero-cost Retrieval-Augmented Generation (RAG) pipeline built as a personal AI profiling system. No OpenAI. No API keys. Runs entirely on your machine using ChromaDB, HuggingFace embeddings, and Ollama.

Built as a portfolio project covering **Production RAG** and **Monitoring & Observability** — from chunking strategies through hybrid retrieval, reranking, citation enforcement, RAGAS evaluation, and a CI-gated quality gate.

---

## What It Does

Given a set of personal documents, the pipeline answers questions about the person by:
1. Retrieving relevant chunks using hybrid search (BM25 + vector) fused with RRF
2. Reranking results with a cross-encoder for precision
3. Generating concise, cited answers using a local LLM (qwen2.5:1.5b via Ollama)
4. Evaluating answer quality across 6 metrics using RAGAS + custom citation scoring
5. Gating the pipeline on quality thresholds — fails CI if any metric regresses

---

## Stack

| Component | Tool |
|---|---|
| Vector store | ChromaDB |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace) |
| Sparse retrieval | BM25 (LangChain) |
| Reranking | Cross-encoder (`ms-marco-MiniLM-L-6-v2`) |
| LLM | `qwen2.5:1.5b` via Ollama |
| Evaluation | RAGAS + custom citation metrics |
| Language | Python 3.11 |

All models run locally. No internet required after initial setup.

---

## Project Structure

```
RAG/
├── docs/                        # Source documents (plain .txt)
├── db/
│   └── chroma_db/               # Persisted ChromaDB vector store
│
├── 01_Ingestion.py              # Document loading and vector store ingestion
├── 02_retrieval.py              # Basic vector similarity retrieval
├── 03_answer.py                 # Answer generation with retrieved context
├── 04_history.py                # Conversation history / multi-turn support
├── 05_recursive_chunking.py     # RecursiveCharacterTextSplitter chunking
├── 06_semantic_chunking.py      # Embedding-based semantic chunking
├── 07_multi_query.py            # Query expansion for improved recall
├── 08_reciprocal_rank.py        # Reciprocal Rank Fusion (RRF)
├── 09_hybrid_search.py          # BM25 + vector hybrid search
├── 10_reranker.py               # Cross-encoder reranking
│
├── 11_ragas_eval.py             # RAGAS baseline evaluation
├── 12_citation_eval.py          # Citation enforcement + coverage scoring
├── 13_ci_gate.py                # CI quality gate (exits 1 on regression)
│
├── ragas_results.json           # Run 1 raw scores
├── citation_ragas_results.json  # Run 3 raw scores
├── ci_gate_report.json          # Latest CI gate report
└── requirements.txt
```

---

## Setup

### Prerequisites
- Python 3.11
- [Ollama](https://ollama.com) installed and running
- `qwen2.5:1.5b` model pulled

```bash
ollama pull qwen2.5:1.5b
```

### Install dependencies

```bash
git clone https://github.com/kabiryadav0932/RAG.git
cd RAG
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Add your documents

Place `.txt` files in the `docs/` directory. Then build the vector store:

```bash
python 01_Ingestion.py   # loads docs/, chunks, and builds the ChromaDB vector store
```

---

## Running the Pipeline

### Full evaluation (RAGAS + citation metrics)
```bash
python 12_citation_eval.py
```

### CI quality gate
```bash
python 13_ci_gate.py
# exits 0 if all checks pass, exits 1 on regression
```

---

## Evaluation Results

Three evaluation runs were conducted to measure the effect of prompt engineering and citation enforcement on answer quality.

### Metric definitions

| Metric | What it measures |
|---|---|
| **Faithfulness** | Is the answer grounded in retrieved context? (no hallucination) |
| **Answer relevancy** | Does the answer directly address the question? |
| **Context precision** | Were the retrieved chunks actually useful? |
| **Context recall** | Did retrieval surface everything needed to answer? |
| **Citation coverage** | What % of answers contain at least one `[Source N]` citation? |
| **Citation precision** | What % of cited sources actually exist in the retrieved set? |

### Scores across runs

| Metric | Run 1 (baseline) | Run 2 (tight prompt) | Run 3 (citations) |
|---|---|---|---|
| Faithfulness | 0.8750 | 0.7750 | **0.8500** |
| Answer relevancy | 0.5215 | 0.7071 | **0.7330** |
| Context precision | 0.9583 | 0.9583 | **1.0000** |
| Context recall | 1.0000 | 1.0000 | **1.0000** |
| Citation coverage | — | — | **0.8000** |
| Citation precision | — | — | **1.0000** |
| **Overall** | **0.8387** | **0.8601** | **0.8601** |

### Key observations

- **Answer relevancy improved +40% from run 1 → run 3** (0.52 → 0.73) purely through prompt engineering — no model swap, no retraining
- **Citation precision is perfect (1.0)** — every citation produced by the model points to a valid retrieved source
- **Faithfulness vs relevancy trade-off is documented**: tighter prompts improve relevancy but can reduce faithfulness as the model summarises more aggressively. This is a known RAG tension.
- **Context recall is perfect across all runs** — the hybrid BM25 + vector retriever surfaces all necessary information consistently

### CI gate thresholds (run 3 baseline)

```python
THRESHOLDS = {
    "faithfulness":       0.70,
    "answer_relevancy":   0.65,
    "context_precision":  0.90,
    "context_recall":     0.95,
    "citation_coverage":  0.70,
    "citation_precision": 0.80,
}
```

All 6 checks pass on the current pipeline. Any regression below these thresholds causes `13_ci_gate.py` to exit with code 1.

---

## Retrieval Architecture

```
Query
  │
  ├─── Vector search (ChromaDB + MiniLM-L6-v2)  ──┐
  │                                                 ├── RRF fusion → Top-K chunks
  └─── BM25 sparse search                        ──┘
                                                     │
                                              Cross-encoder rerank
                                                     │
                                              LLM generation (Ollama)
                                                     │
                                              Answer + [Source N] citations
```

---

## Design Decisions

**Why fully local?** Privacy, zero cost, and to demonstrate understanding of inference constraints — the same trade-offs that matter in real enterprise deployments.

**Why qwen2.5:1.5b?** Fastest local model available on CPU hardware (2s/query vs 20s for llama3.2). Sufficient for short factual extraction tasks.

**Why hybrid retrieval + RRF?** BM25 handles exact keyword matches; vector search handles semantic similarity. RRF fuses both rank lists without requiring score normalisation. This combination consistently outperforms either retriever alone on recall.

**Why RAGAS for evaluation?** RAGAS provides decomposed metrics that isolate retrieval quality from generation quality — critical for debugging whether a bad answer comes from the retriever or the LLM.

---

## Requirements

```
langchain
langchain-chroma
langchain-huggingface
langchain-community
langchain-ollama
langchain-text-splitters
chromadb
sentence-transformers
ragas
datasets
rank-bm25
```

---

## Author

Kabir — 3rd year CSE, PES University Bangalore  
GitHub: [@kabiryadav0932](https://github.com/kabiryadav0932)
