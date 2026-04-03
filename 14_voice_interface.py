"""
15_voice_interface.py
---------------------
Voice interface for the RAG pipeline.

  🎤  Speak  →  Whisper (STT)  →  RAG pipeline  →  Piper (TTS)  →  🔊 Hear

How to use:
  - Press Enter to start recording
  - Speak your question
  - Silence for 2 seconds stops the recording automatically
  - The answer is spoken back to you
  - Press Ctrl+C to quit

Run:
  python 15_voice_interface.py
"""

import wave
import tempfile
import warnings
import numpy as np
import sounddevice as sd
warnings.filterwarnings("ignore", category=DeprecationWarning)

from pathlib import Path
from faster_whisper import WhisperModel
from piper.voice import PiperVoice
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM

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

WHISPER_MODEL_SIZE = "base"          # tiny/base/small — base is best balance on CPU
PIPER_MODEL        = "models/piper/en_US-lessac-medium.onnx"
PIPER_CONFIG       = "models/piper/en_US-lessac-medium.onnx.json"

MIC_DEVICE         = 0               # sof-hda-dsp hw:0,0 — 2 inputs
SAMPLE_RATE        = 16000           # Whisper expects 16kHz


# ── Load RAG pipeline ─────────────────────────────────────────────────────────
def load_pipeline():
    print("[init] Loading embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("[init] Loading vector store...")
    vectorstore = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )

    print("[init] Loading BM25 retriever...")
    loader   = DirectoryLoader(DOCS_PATH, glob="**/*.txt", loader_cls=TextLoader)
    docs     = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=BM25_CHUNK_SIZE,
        chunk_overlap=BM25_CHUNK_OVERLAP,
    )
    chunks         = splitter.split_documents(docs)
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = TOP_K
    print(f"[init] BM25 indexed {len(chunks)} chunks")

    print(f"[init] Loading LLM ({OLLAMA_MODEL})...")
    llm = OllamaLLM(model=OLLAMA_MODEL)

    return vectorstore, bm25_retriever, llm

# ── Load Whisper ──────────────────────────────────────────────────────────────
def load_whisper():
    print(f"[init] Loading Whisper ({WHISPER_MODEL_SIZE})...")
    # compute_type="int8" is fastest on CPU
    model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    print("[init] Whisper ready.")
    return model

# ── Load Piper ────────────────────────────────────────────────────────────────
def load_piper():
    print("[init] Loading Piper TTS voice...")
    if not Path(PIPER_MODEL).exists():
        raise FileNotFoundError(
            f"Piper model not found at {PIPER_MODEL}\n"
            "Run the wget commands from the setup instructions first."
        )
    voice = PiperVoice.load(PIPER_MODEL, config_path=PIPER_CONFIG, use_cuda=False)
    print("[init] Piper ready.")
    return voice

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

# ── Hybrid retrieve ───────────────────────────────────────────────────────────
def hybrid_retrieve(query, vectorstore, bm25_retriever):
    vector_results = vectorstore.similarity_search(query, k=TOP_K)
    bm25_results   = bm25_retriever.invoke(query)
    fused          = reciprocal_rank_fusion([vector_results, bm25_results])
    return fused[:TOP_K]

# ── Generate answer ───────────────────────────────────────────────────────────
def generate_answer(question, docs, llm):
    context = "\n\n---\n\n".join(
        f"[Source {i+1}]: {doc.page_content}"
        for i, doc in enumerate(docs)
    )
    prompt = f"""You are a precise fact extraction assistant.
Answer the question in 1-2 sentences using ONLY the context below.
Keep the answer SHORT and natural — it will be read aloud.
Do NOT use bullet points, markdown, or [Source N] citations in this response.
If the answer is not in the context, say: "I don't know based on my documents."

Context:
{context}

Question: {question}

Answer:"""

    answer = llm.invoke(prompt)
    # Trim to 2 sentences max
    sentences = [s.strip() for s in answer.strip().split('.') if s.strip()]
    trimmed   = '. '.join(sentences[:2])
    return trimmed + '.' if trimmed and not trimmed.endswith('.') else trimmed

# ── Record audio with auto-stop on silence ────────────────────────────────────
def record_manual():
    import threading

    frames = []

    def callback(indata, frame_count, time_info, status):
        frames.append(indata[:, 0].copy())

    print("[🎤] Recording... (press Enter to stop)")

    stream = sd.InputStream(
        device=MIC_DEVICE,
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=callback,
        blocksize=1024,
    )
    stream.start()
    input()          # blocks until user presses Enter
    stream.stop()
    stream.close()

    audio = np.concatenate(frames, axis=0)
    print(f"[🎤] Recorded {len(audio)/SAMPLE_RATE:.1f}s of audio.")
    return audio

# ── Transcribe with Whisper ───────────────────────────────────────────────────
def transcribe(audio, whisper_model):
    # Save to a temp wav file — faster-whisper needs a file path or numpy array
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
        with wave.open(f, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(SAMPLE_RATE)
            pcm = (audio * 32767).astype(np.int16)
            wf.writeframes(pcm.tobytes())

    segments, info = whisper_model.transcribe(tmp_path, language="en", beam_size=5)
    text = " ".join(seg.text for seg in segments).strip()
    Path(tmp_path).unlink(missing_ok=True)
    return text

# ── Speak with Piper ──────────────────────────────────────────────────────────
def speak(text, piper_voice):
    print("[🔊] Speaking...")
    # Piper requires a real file — BytesIO doesn't set channels correctly
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name

    with wave.open(tmp_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(22050)   # Piper lessac-medium outputs 22050Hz
        piper_voice.synthesize(text, wf)

    with wave.open(tmp_path, 'rb') as wf:
        sample_rate = wf.getframerate()
        frames      = wf.readframes(wf.getnframes())

    Path(tmp_path).unlink(missing_ok=True)

    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    sd.play(audio, samplerate=sample_rate)
    sd.wait()

# ── Main loop ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Another Me — Voice Interface")
    print("=" * 60)

    # Load all models once
    vectorstore, bm25_retriever, llm = load_pipeline()
    whisper_model = load_whisper()
    piper_voice   = load_piper()

    print("\n" + "=" * 60)
    print("  Press Enter to START recording.")
    print("  Press Enter again to STOP and get your answer.")
    print("  Press Ctrl+C to quit.")
    print("=" * 60)

    while True:
        try:
            input("\n[Press Enter to start speaking]")

            # 1. Record until second Enter
            audio = record_manual()
            if len(audio) / SAMPLE_RATE < 0.5:
                print("[!] Too short — try again.")
                continue

            # 2. Transcribe
            print("[⏳] Transcribing...")
            question = transcribe(audio, whisper_model)
            if not question.strip():
                print("[!] Couldn't hear anything — try again.")
                continue
            print(f"\n[You]  {question}")

            # 3. RAG retrieve + generate
            print("[⏳] Thinking...")
            docs   = hybrid_retrieve(question, vectorstore, bm25_retriever)
            answer = generate_answer(question, docs, llm)
            print(f"[Me]   {answer}")

            # 4. Speak answer
            speak(answer, piper_voice)

        except KeyboardInterrupt:
            print("\n\n[Bye!]")
            break
        except Exception as e:
            print(f"[error] {e}")
            continue