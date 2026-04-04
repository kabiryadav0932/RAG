# Another Me — Local RAG Pipeline + Voice Interface

A fully local, zero-cost Retrieval-Augmented Generation (RAG) pipeline built as a personal AI profiling system. No OpenAI. No API keys. Runs entirely on your machine using ChromaDB, HuggingFace embeddings, Ollama, and a full voice interface powered by Whisper STT and Kokoro TTS.

Built as a portfolio project covering **Production RAG**, **Monitoring & Observability**, and **Voice AI** — from chunking strategies through hybrid retrieval, reranking, citation enforcement, RAGAS evaluation, a CI-gated quality gate, and a fully local voice assistant you can speak to and hear back from.

---

## What It Does

Given a set of personal documents, the pipeline answers questions about the person by:
1. Retrieving relevant chunks using hybrid search (BM25 + vector) fused with RRF
2. Reranking results with a cross-encoder for precision
3. Generating concise answers using a local LLM (llama3.2 via Ollama)
4. Maintaining **conversation memory** across turns within a session
5. Evaluating answer quality across 6 metrics using RAGAS + custom citation scoring
6. Gating the pipeline on quality thresholds — fails CI if any metric regresses
7. **Speaking answers aloud** using Kokoro neural TTS, and **listening to questions** via Whisper STT
8. **Saving session history** to `docs/history.txt` automatically on exit
9. **Hybrid question mode** — answers personal questions from your docs and general questions from world knowledge

---

## Stack

| Component | Tool |
|---|---|
| Vector store | ChromaDB |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace) |
| Sparse retrieval | BM25 (LangChain) |
| Reranking | Cross-encoder (`ms-marco-MiniLM-L-6-v2`) |
| LLM | `llama3.2` via Ollama |
| Evaluation | RAGAS + custom citation metrics |
| Speech-to-text | `faster-whisper` (small model, CPU) |
| Text-to-speech | Kokoro ONNX (`af_sky` voice) |
| Stop word | `"ok"` — say it at the end of your question to stop recording |
| Audio I/O | `sounddevice` + `scipy` (resampling) |
| Language | Python 3.11 |

All models run locally. No internet required after initial setup.

---

## Project Structure

```
RAG/
├── docs/                        # Source documents (plain .txt)
│   ├── data.txt                 # Personal knowledge base
│   ├── personal.txt             # Additional personal facts
│   └── history.txt              # Auto-saved session conversation logs
├── db/
│   └── chroma_db/               # Persisted ChromaDB vector store
├── models/
│   └── kokoro/                  # Kokoro TTS model files
│       ├── kokoro-v0_19.onnx
│       └── voices.bin
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
├── 14_voice_interface.py        # Voice interface — speak, hear, save history
├── update.py                    # CLI tool to add new facts and re-ingest
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
- `llama3.2` model pulled

```bash
ollama pull llama3.2
```

### Install dependencies

```bash
git clone https://github.com/kabiryadav0932/RAG.git
cd RAG
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install faster-whisper kokoro-onnx sounddevice scipy soundfile
```

### Add your documents

Place `.txt` files in the `docs/` directory. Then build the vector store:

```bash
python 01_Ingestion.py   # loads docs/, chunks, and builds the ChromaDB vector store
```

### Download Kokoro TTS models

```bash
mkdir -p models/kokoro
cd models/kokoro
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.bin
cd ../..
```

---

## Running the Pipeline

### Voice interface
```bash
python 14_voice_interface.py
```
- Starts immediately — no wake word needed
- Speak your question and say **"ok"** at the end to stop recording
- The answer is spoken back in a natural voice (`af_sky`)
- Conversation memory is maintained across turns within the session
- Session is automatically saved to `docs/history.txt` on exit
- Press **Ctrl+C** to quit

### Update your knowledge base
```bash
# Add a single fact (appends to data.txt + re-ingests automatically)
python update.py "I started learning Japanese today."

# Add without re-ingesting (batch multiple facts first)
python update.py "My new hobby is photography." --no-reingest

# Bulk import from a file (one fact per line)
python update.py --file my_notes.txt

# View recent entries
python update.py --list
```

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

## Voice Interface Architecture

```
🎤 Microphone (48000 Hz)
       │
  Resample → 16000 Hz
       │
  Whisper STT (faster-whisper, small, CPU)
  [say "ok" to stop recording]
       │
  Question text
       │
  ┌────────────────────────────────────┐
  │  Hybrid Question Router (prompt)   │
  │  PERSONAL → RAG docs               │
  │  GENERAL  → LLM world knowledge    │
  └────────────────────────────────────┘
       │
  Conversation history (last 3 exchanges)
       │
  RAG Pipeline (BM25 + vector + RRF)
       │
  Answer text
       │
  Kokoro TTS (ONNX, af_sky)
       │
  Resample 24000 → 48000 Hz
       │
🔊 Speaker
       │
  Append to docs/history.txt on exit
```

**Stop word:** Say "ok" at the end of your question to stop recording. The watcher model (Whisper tiny) runs on a rolling buffer in a background thread, checking every 1.5 seconds for "ok".

**Hybrid mode:** The LLM is prompted with two explicit modes — PERSONAL (uses RAG context) and GENERAL (uses world knowledge). This means "What is my favourite movie?" uses your docs, while "Tell me about Barcelona FC" gets a real football answer without hallucinating a connection to you.

**Session history:** Every conversation is appended to `docs/history.txt` with a timestamp header when you press Ctrl+C. The file is created automatically if it doesn't exist. It is not indexed into ChromaDB — it's purely for your own reference.

**Conversation memory:** The last 3 exchanges (6 turns) are injected into the prompt each turn, so follow-up questions like "why do I like it?" work naturally without repeating context.

**Audio resampling:** Your ALSA device runs at 48000 Hz natively. All resampling (mic input and speaker output) uses `scipy.signal.resample_poly` for artifact-free audio.

---

## Session History Format

Each session is appended to `docs/history.txt` in this format:

```
============================================================
Session: 2026-04-04 18:12:30
============================================================
[You]: What is your favourite movie?
[Me]: My all-time favorite movie is Veer-Zaara.
[You]: Do I exercise?
[Me]: Yes, I exercise for about 5–10 minutes daily.
```

---

## Knowledge Base Management

"Another Me" is designed to grow with you. Use `update.py` to teach it new facts at any time:

```bash
python update.py "I got selected for a research internship at IISc."
python update.py "My CGPA this semester is 9.1."
python update.py "I finished reading Atomic Habits."
```

Each fact is timestamped and appended to `docs/data.txt`. The vector store is automatically rebuilt so the new information is immediately queryable via the voice interface.

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
                                              LLM generation (llama3.2)
                                                     │
                                              Answer + [Source N] citations
```

---

## Design Decisions

**Why fully local?** Privacy, zero cost, and to demonstrate understanding of inference constraints — the same trade-offs that matter in real enterprise deployments.

**Why llama3.2?** Better instruction following and significantly reduced hallucination compared to smaller models.

**Why hybrid retrieval + RRF?** BM25 handles exact keyword matches; vector search handles semantic similarity. RRF fuses both rank lists without requiring score normalisation. This combination consistently outperforms either retriever alone on recall.

**Why hybrid question routing via prompt?** An earlier version used a wake word ("Okiro") and an exit word ("Sayonara") for session control. These were removed — the wake word added friction with no benefit in a single-user setup, and the exit word was unreliable with STT. Ctrl+C is cleaner. The routing logic (personal vs general) is now handled entirely in the LLM prompt, which proved more reliable than a classifier.

**Why Whisper small for transcription?** The base model produced too many mishearing errors on conversational speech. Small gives meaningfully better accuracy with acceptable latency on an i5 CPU (~2-3s per transcription).

**Why Whisper tiny for "ok" detection?** Speed over accuracy — the stop word watcher runs every 1.5 seconds on a rolling buffer. Tiny is fast enough to not lag the recording loop.

**Why Kokoro TTS over Piper?** Kokoro uses a neural ONNX model producing natural-sounding speech. Piper's `synthesize()` produced empty WAV output on this hardware configuration. Kokoro worked out of the box with no issues.

**Why save history to docs/history.txt but not index it?** History is for the user's reference only. Indexing it into ChromaDB would pollute the personal knowledge base with conversational noise and degrade retrieval quality.

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
faster-whisper
kokoro-onnx
sounddevice
scipy
soundfile
```

---

## Author

Kabir — 3rd year CSE, PES University Bangalore  
GitHub: [@kabiryadav0932](https://github.com/kabiryadav0932)
