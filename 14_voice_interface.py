"""
14_voice_interface.py
---------------------
Voice interface for the RAG pipeline.

  Speak → Whisper (STT) → RAG pipeline → Kokoro (TTS) → 🔊 Hear

How to use:
  - Speak your question and say "ok" at the end to stop recording
  - The answer is spoken back to you — then it's ready for your next question immediately
  - Conversation memory is maintained across turns within the session
  - Press Ctrl+C to quit

Run:
  python 14_voice_interface.py
"""

import wave
import threading
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
HISTORY_FILE       = "docs/history.txt"
DB_PATH            = "db/chroma_db"
EMBEDDING_MODEL    = "all-MiniLM-L6-v2"
COLLECTION_NAME    = "main_chunks"
OLLAMA_MODEL       = "llama3.2"
TOP_K              = 3
RRF_K              = 60
BM25_CHUNK_SIZE    = 500
BM25_CHUNK_OVERLAP = 50

WHISPER_MODEL_SIZE      = "small"     # used for transcription — best accuracy on CPU
WHISPER_WATCHER_SIZE    = "tiny"     # used only for "ok" detection — speed over accuracy
KOKORO_MODEL       = "models/kokoro/kokoro-v0_19.onnx"
KOKORO_VOICES      = "models/kokoro/voices.bin"
KOKORO_VOICE       = "af_sky"

MIC_DEVICE         = 0               # sof-hda-dsp hw:0,0 — 2 inputs
DEVICE_RATE        = 48000           # native rate for sof-hda-dsp
WHISPER_RATE       = 16000           # Whisper expects 16kHz

NE_STOP_WORD       = "ok"           # ends your question hands-free

# How often (in seconds) to check the rolling buffer for "ok" while recording
NE_CHECK_INTERVAL  = 1.5            # check every 1.5s of new audio

# VAD — Voice Activity Detection
VAD_RMS_THRESHOLD  = 0.02           # RMS energy below this = silence (tune if needed)
VAD_SILENCE_RATIO  = 0.85           # if >85% of frames are silent, skip the chunk


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
    print(f"[init] Loading Whisper ({WHISPER_MODEL_SIZE}) for transcription...")
    model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    print("[init] Whisper base ready.")
    print(f"[init] Loading Whisper ({WHISPER_WATCHER_SIZE}) for 'ok' detection...")
    watcher_model = WhisperModel(WHISPER_WATCHER_SIZE, device="cpu", compute_type="int8")
    print("[init] Whisper tiny ready.")
    return model, watcher_model

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

    prompt = f"""You are "Another Me" — a personal AI with two modes:
1. PERSONAL: If the question is about the person (their life, preferences, background, work, opinions), answer using the Context and conversation history below.
2. GENERAL: If the question is about the world (sports, science, history, people, places, facts), answer directly from your general knowledge — do NOT try to connect it to the person.

Answer in 1-2 sentences. Keep it SHORT and natural for speech.
Do NOT use bullet points, markdown, or citations.

Context (personal documents):
{context}

{"Conversation so far:" + chr(10) + history_text + chr(10) if history_text else ""}
Question: {question}

Answer:"""

    answer = llm.invoke(prompt)
    sentences = [s.strip() for s in answer.strip().split('.') if s.strip()]
    trimmed   = '. '.join(sentences[:2])
    return trimmed + '.' if trimmed and not trimmed.endswith('.') else trimmed

# ── Quick transcribe helper (for "ne" detection) ──────────────────────────────
def _quick_transcribe(audio_chunk, whisper_model):
    """Transcribe a short audio chunk. Returns lowercase text."""
    from math import gcd
    g = gcd(DEVICE_RATE, WHISPER_RATE)
    audio_16k = sps.resample_poly(audio_chunk, WHISPER_RATE // g, DEVICE_RATE // g)

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
        beam_size=1,
        initial_prompt="ok.",
    )
    text = " ".join(seg.text for seg in segments).strip().lower()
    Path(tmp_path).unlink(missing_ok=True)
    return text

# ── Record audio — stops when "ne" is heard ───────────────────────────────────
def record_until_ne(whisper_model, watcher_model):
    """
    Records audio continuously. Every NE_CHECK_INTERVAL seconds, the latest
    chunk is transcribed to check for "ne". When detected, recording stops.
    Returns the full recorded audio (resampled to WHISPER_RATE).
    """
    from math import gcd

    frames      = []          # all recorded frames
    stop_event  = threading.Event()
    ne_detected = threading.Event()

    check_chunk_samples = int(DEVICE_RATE * NE_CHECK_INTERVAL)

    def callback(indata, frame_count, time_info, status):
        frames.append(indata[:, 0].copy())

    stream = sd.InputStream(
        device=MIC_DEVICE,
        samplerate=DEVICE_RATE,
        channels=1,
        dtype="float32",
        callback=callback,
        blocksize=1024,
    )

    print("[🎤] Recording... (say 'ok' to stop)")
    stream.start()

    # Background thread: periodically check last chunk for "ne"
    def ne_watcher():
        last_checked = 0
        while not stop_event.is_set():
            # Collect enough new samples before checking
            current = sum(len(f) for f in frames)
            new_samples = current - last_checked
            if new_samples >= check_chunk_samples:
                # Grab the last check_chunk_samples worth of audio
                all_audio = np.concatenate(frames, axis=0)
                chunk     = all_audio[-check_chunk_samples:]
                last_checked = current

                # VAD: skip if chunk is mostly silence
                frame_size = 512
                frames_vad = [chunk[i:i+frame_size] for i in range(0, len(chunk), frame_size)]
                silent_frames = sum(1 for f in frames_vad if np.sqrt(np.mean(f**2)) < VAD_RMS_THRESHOLD)
                if silent_frames / max(len(frames_vad), 1) > VAD_SILENCE_RATIO:
                    continue

                text = _quick_transcribe(chunk, watcher_model)
                # "ne" often transcribed as "ne", "neh", "ね", or at end of sentence
                words = text.strip().rstrip(".!?,").split()
                if words and words[-1].lower() in ("ok", "okay", "oké"):
                    print(f"[OK] Stop word detected! ({text})")
                    ne_detected.set()
                    stop_event.set()

            stop_event.wait(timeout=0.1)  # small sleep to avoid busy-loop

    watcher_thread = threading.Thread(target=ne_watcher, daemon=True)
    watcher_thread.start()

    # Wait until "ne" is detected
    ne_detected.wait()

    stop_event.set()
    stream.stop()
    stream.close()
    watcher_thread.join(timeout=2)

    audio = np.concatenate(frames, axis=0)
    duration = len(audio) / DEVICE_RATE
    print(f"[🎤] Recorded {duration:.1f}s of audio.")

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

# ── Save session history ──────────────────────────────────────────────────────
def save_history(chat_history):
    if not chat_history:
        return
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Session: {timestamp}\n")
        f.write(f"{'='*60}\n")
        for role, text in chat_history:
            label = "You" if role == "user" else "Me"
            f.write(f"[{label}]: {text}\n")
    print(f"[💾] Session saved to {HISTORY_FILE}")

# ── Main loop ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Another Me — Voice Interface")
    print("=" * 60)

    vectorstore, bm25_retriever, llm = load_pipeline()
    whisper_model, watcher_model = load_whisper()
    kokoro        = load_kokoro()

    print("\n" + "=" * 60)
    print(f"  Speak your question and end with 'ok' to stop recording.")
    print("  Press Ctrl+C to quit.")
    print("=" * 60)

    chat_history = []  # in-memory conversation history

    # ── Greet immediately ─────────────────────────────────────────────────────
    try:
        speak("Ohayou. I'm ready. Ask me anything.", kokoro)
    except KeyboardInterrupt:
        save_history(chat_history)
        print("\n\n[Sayonara!]")
        exit(0)

    # ── Conversation loop ─────────────────────────────────────────────────────
    while True:
        try:
            # 1. Record question — stops when "ok" is heard
            audio = record_until_ne(whisper_model, watcher_model)
            if len(audio) / WHISPER_RATE < 0.5:
                print("[!] Too short — try again.")
                continue

            # 2. Transcribe
            print("[⏳] Transcribing...")
            question = transcribe(audio, whisper_model)
            if not question.strip():
                print("[!] Couldn't hear anything — try again.")
                continue

            # Strip trailing "ok" from the transcribed question
            words = question.strip().rstrip(".!?,").split()
            if words and words[-1].lower() in ("ok", "okay", "oké", "k"):
                question = " ".join(words[:-1]).strip()

            print(f"\n[You]  {question}")

            # 3. RAG retrieve + generate
            print("[⏳] Thinking...")
            docs   = hybrid_retrieve(question, vectorstore, bm25_retriever)
            answer = generate_answer(question, docs, llm, chat_history)
            print(f"[Me]   {answer}")

            # 4. Update chat history
            chat_history.append(("user", question))
            chat_history.append(("assistant", answer))

            # 5. Speak answer
            speak(answer, kokoro)

        except KeyboardInterrupt:
            save_history(chat_history)
            print("\n\n[Sayonara!]")
            break
        except Exception as e:
            print(f"[error] {e}")
            continue
