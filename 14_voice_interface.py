"""
14_voice_interface.py
---------------------
Voice interface for the RAG pipeline.

  Say "Okiro" → Whisper (STT) → RAG pipeline → Kokoro (TTS) → 🔊 Hear

How to use:
  - Say "Okiro" (Japanese for "wake up") to activate
  - Speak your question
  - Press Enter to stop recording
  - The answer is spoken back to you
  - Conversation memory is maintained across turns within the session
  - Press Ctrl+C to quit

Run:
  python 14_voice_interface.py
"""

import wave
import tempfile
import warnings
import numpy as np
import sounddevice as sd
import scipy.signal as sps
warnings.filterwarnings("ignore", category=DeprecationWarning)

from pathlib import Path
from faster_whisper import WhisperModel
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from kokoro_onnx import Kokoro

# ── Config ────────────────────────────────────────────────────────────────────
DOCS_PATH          = "docs"
DB_PATH            = "db/chroma_db"
EMBEDDING_MODEL    = "all-MiniLM-L6-v2"
COLLECTION_NAME    = "main_chunks"
OLLAMA_MODEL       = "llama3.2"
TOP_K              = 3
RRF_K              = 60
BM25_CHUNK_SIZE    = 500
BM25_CHUNK_OVERLAP = 50

WHISPER_MODEL_SIZE = "base"          # tiny/base/small — base is best balance on CPU
KOKORO_MODEL       = "models/kokoro/kokoro-v0_19.onnx"
KOKORO_VOICES      = "models/kokoro/voices.bin"
KOKORO_VOICE       = "af_bella"

MIC_DEVICE         = 0               # sof-hda-dsp hw:0,0 — 2 inputs
DEVICE_RATE        = 48000           # native rate for sof-hda-dsp
WHISPER_RATE       = 16000           # Whisper expects 16kHz

WAKE_WORD          = "okiro"         # Japanese for "wake up"
WAKE_CHUNK_SECS    = 2.5             # seconds of audio per wake-word check


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
    model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    print("[init] Whisper ready.")
    return model

# ── Load Kokoro ───────────────────────────────────────────────────────────────
def load_kokoro():
    print("[init] Loading Kokoro TTS...")
    kokoro = Kokoro(KOKORO_MODEL, KOKORO_VOICES)
    print("[init] Kokoro ready.")
    return kokoro

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

# ── Wake word listener ────────────────────────────────────────────────────────
def listen_for_wake_word(whisper_model):
    """Continuously listen in short chunks until wake word is detected."""
    from math import gcd
    chunk_samples = int(DEVICE_RATE * WAKE_CHUNK_SECS)
    g = gcd(DEVICE_RATE, WHISPER_RATE)

    print(f"[👂] Listening for wake word '{WAKE_WORD}'...")

    while True:
        audio = sd.rec(
            chunk_samples,
            samplerate=DEVICE_RATE,
            channels=1,
            dtype="float32",
            device=MIC_DEVICE,
        )
        sd.wait()
        audio = audio[:, 0]

        # Skip if too quiet (silence)
        if np.abs(audio).max() < 0.01:
            continue

        # Resample to 16kHz for Whisper
        audio_16k = sps.resample_poly(audio, WHISPER_RATE // g, DEVICE_RATE // g)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
            with wave.open(f, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(WHISPER_RATE)
                pcm = (audio_16k * 32767).astype(np.int16)
                wf.writeframes(pcm.tobytes())

        segments, _ = whisper_model.transcribe(
            tmp_path,
            language="en",
            beam_size=1,          # fast — just need to catch the wake word
            initial_prompt="Okiro.",
        )
        text = " ".join(seg.text for seg in segments).strip().lower()
        Path(tmp_path).unlink(missing_ok=True)

        if WAKE_WORD in text:
            print(f"[✅] Wake word detected! ({text})")
            return

# ── Generate answer ───────────────────────────────────────────────────────────
def generate_answer(question, docs, llm, chat_history):
    context = "\n\n---\n\n".join(
        f"[Source {i+1}]: {doc.page_content}"
        for i, doc in enumerate(docs)
    )

    # Format last 3 exchanges (6 turns) of history
    history_text = ""
    if chat_history:
        lines = []
        for role, text in chat_history[-6:]:
            label = "User" if role == "user" else "Assistant"
            lines.append(f"{label}: {text}")
        history_text = "\n".join(lines)

    prompt = f"""You are "Another Me" — a personal AI that answers questions about a specific person using their documents.
Answer in 1-2 sentences. Keep it SHORT and natural for speech.
Do NOT use bullet points, markdown, or citations.
If the answer is not in the context, use the conversation history or your general knowledge to help.

Context:
{context}

{"Conversation so far:" + chr(10) + history_text + chr(10) if history_text else ""}
Question: {question}

Answer:"""

    answer = llm.invoke(prompt)
    sentences = [s.strip() for s in answer.strip().split('.') if s.strip()]
    trimmed   = '. '.join(sentences[:2])
    return trimmed + '.' if trimmed and not trimmed.endswith('.') else trimmed

# ── Record audio ──────────────────────────────────────────────────────────────
def record_manual():
    frames = []

    def callback(indata, frame_count, time_info, status):
        frames.append(indata[:, 0].copy())

    print("[🎤] Recording... (press Enter to stop)")

    stream = sd.InputStream(
        device=MIC_DEVICE,
        samplerate=DEVICE_RATE,
        channels=1,
        dtype="float32",
        callback=callback,
        blocksize=1024,
    )
    stream.start()
    input()
    stream.stop()
    stream.close()

    audio = np.concatenate(frames, axis=0)
    print(f"[🎤] Recorded {len(audio)/DEVICE_RATE:.1f}s of audio.")

    from math import gcd
    g = gcd(DEVICE_RATE, WHISPER_RATE)
    audio = sps.resample_poly(audio, WHISPER_RATE // g, DEVICE_RATE // g)

    return audio

# ── Transcribe with Whisper ───────────────────────────────────────────────────
def transcribe(audio, whisper_model):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
        with wave.open(f, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(WHISPER_RATE)
            pcm = (audio * 32767).astype(np.int16)
            wf.writeframes(pcm.tobytes())

    segments, info = whisper_model.transcribe(
        tmp_path,
        language="en",
        beam_size=5,
        initial_prompt="Kabir, PES University, Bangalore, Veer-Zaara, Bollywood, India, Bihar, Patna."
    )
    text = " ".join(seg.text for seg in segments).strip()
    Path(tmp_path).unlink(missing_ok=True)
    return text

# ── Speak with Kokoro ─────────────────────────────────────────────────────────
def speak(text, kokoro):
    print("[🔊] Speaking...")
    from math import gcd
    samples, rate = kokoro.create(text, voice=KOKORO_VOICE, speed=1.0, lang="en-us")
    g = gcd(rate, DEVICE_RATE)
    audio = sps.resample_poly(samples, DEVICE_RATE // g, rate // g)
    sd.play(audio, samplerate=DEVICE_RATE)
    sd.wait()

# ── Main loop ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Another Me — Voice Interface")
    print("=" * 60)

    vectorstore, bm25_retriever, llm = load_pipeline()
    whisper_model = load_whisper()
    kokoro        = load_kokoro()

    print("\n" + "=" * 60)
    print(f"  Say '{WAKE_WORD}' to start speaking.")
    print("  Then press Enter to STOP and get your answer.")
    print("  Press Ctrl+C to quit.")
    print("=" * 60)

    chat_history = []  # in-memory conversation history

    while True:
        try:
            # 1. Wait for wake word
            listen_for_wake_word(whisper_model)

            # 2. Record question until Enter
            audio = record_manual()
            if len(audio) / WHISPER_RATE < 0.5:
                print("[!] Too short — try again.")
                continue

            # 3. Transcribe
            print("[⏳] Transcribing...")
            question = transcribe(audio, whisper_model)
            if not question.strip():
                print("[!] Couldn't hear anything — try again.")
                continue
            print(f"\n[You]  {question}")

            # 4. RAG retrieve + generate
            print("[⏳] Thinking...")
            docs   = hybrid_retrieve(question, vectorstore, bm25_retriever)
            answer = generate_answer(question, docs, llm, chat_history)
            print(f"[Me]   {answer}")

            # 5. Update chat history
            chat_history.append(("user", question))
            chat_history.append(("assistant", answer))

            # 6. Speak answer
            speak(answer, kokoro)

        except KeyboardInterrupt:
            print("\n\n[Bye!]")
            break
        except Exception as e:
            print(f"[error] {e}")
            continue
