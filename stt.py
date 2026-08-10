"""Speech-to-Text module — Whisper BPE (primary) + Vosk dictionary (fallback).

Supports all Whisper model sizes and one Vosk fallback model.
Audio capture via PyAudio (preferred) or sounddevice (fallback).
"""

import io
import os
import sys
import json
import struct
import time
import threading
import zipfile
import wave
import tempfile
import urllib.request
from typing import Optional, Tuple, Any

from utils import c, info, success, warn, error, BOLD, CYAN, GREEN, YELLOW, DIM, RESET
from config import LISTEN_TIMEOUT, PHRASE_LIMIT, AMBIENT_DURATION
from normalizer import normalize_utterance

# ─── Optional imports ──────────────────────────────────────────────────────────

try:
    import vosk
    vosk.SetLogLevel(-1)
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False

try:
    import speech_recognition as sr
except ImportError:
    sr = None

try:
    import pyaudio as _pyaudio_check
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

try:
    import sounddevice as sd
    import numpy as np
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False

# ─── Whisper model catalogue ──────────────────────────────────────────────────

WHISPER_MODELS = [
    {"name": "tiny",              "desc": "Tiny (75 MB) — Fastest, basic accuracy"},
    {"name": "tiny.en",           "desc": "Tiny English (75 MB) — Fast, English-only optimized"},
    {"name": "base",              "desc": "Base (150 MB) — Good balance of speed & accuracy"},
    {"name": "base.en",           "desc": "Base English (150 MB) — Fast, English-optimized"},
    {"name": "small",             "desc": "Small (500 MB) — ★ Great accuracy [Recommended]"},
    {"name": "small.en",          "desc": "Small English (500 MB) — ★ Great accuracy, English-only"},
    {"name": "medium",            "desc": "Medium (1.5 GB) — High accuracy, slower"},
    {"name": "medium.en",         "desc": "Medium English (1.5 GB) — High accuracy, English-only"},
    {"name": "large-v3",          "desc": "Large V3 (3 GB) — ★★ Highest accuracy, needs good GPU"},
    {"name": "large-v3-turbo",    "desc": "Large V3 Turbo (1.6 GB) — ★★ Near-best accuracy, faster"},
    {"name": "distil-large-v3",   "desc": "Distil Large V3 (1.5 GB) — Distilled, fast & accurate"},
    {"name": "distil-medium.en",  "desc": "Distil Medium EN (1 GB) — Distilled, English-only"},
    {"name": "distil-small.en",   "desc": "Distil Small EN (500 MB) — Distilled, fast English"},
]

VOSK_FALLBACK_MODEL = {
    "name": "vosk-model-en-us-0.22-lgraph",
    "desc": "Vosk Medium English (128 MB) — Dictionary-based fallback (no GPU needed)",
    "url": "https://alphacephei.com/vosk/models/vosk-model-en-us-0.22-lgraph.zip"
}

# ─── CUDA detection (without requiring torch) ─────────────────────────────────

def _detect_cuda() -> bool:
    """Detect CUDA availability without importing torch."""
    try:
        import ctranslate2
        return "cuda" in ctranslate2.get_supported_compute_types("cuda")
    except Exception:
        pass
    try:
        import subprocess
        result = subprocess.run(["nvidia-smi"], capture_output=True, timeout=3)
        return result.returncode == 0
    except Exception:
        return False


# ─── SpeechListener ───────────────────────────────────────────────────────────

class SpeechListener:
    """Audio capture + transcription engine.

    Primary:  faster-whisper (BPE subword tokenization, GPU accelerated)
    Fallback: Vosk (dictionary-based, CPU only)
    """

    def __init__(self, stt_engine: str = "whisper", whisper_model: str = "small.en",
                 vosk_model_path: Optional[str] = None, enable_rescoring: bool = True):
        self.stt_engine = stt_engine
        self.enable_rescoring = enable_rescoring
        self._context_bias = ""
        self._whisper_model = None
        self._vosk_model = None
        self._use_pyaudio = PYAUDIO_AVAILABLE
        self._sd_energy = 300
        self._calibrated = False
        self._mic_lock = threading.Lock()

        if self._use_pyaudio and sr:
            self.recognizer = sr.Recognizer()
            self.mic = sr.Microphone()
        else:
            self.recognizer = None
            self.mic = None

        self.whisper_model_name = whisper_model
        self.vosk_model_path = vosk_model_path
        self._init_models()

    def _init_models(self):
        """Instantiate STT models (Whisper or Vosk)."""
        # ── Load Whisper if requested and available ─────────────────────────
        if self.stt_engine == "whisper" and FASTER_WHISPER_AVAILABLE:
            has_cuda = _detect_cuda()
            device = "cuda" if has_cuda else "cpu"
            compute_type = "float16" if has_cuda else "int8"
            info(f"Loading Whisper model '{self.whisper_model_name}' on {device} ({compute_type})...")
            try:
                self._whisper_model = WhisperModel(
                    self.whisper_model_name, device=device, compute_type=compute_type
                )
                success(f"Whisper '{self.whisper_model_name}' ready on {device} (BPE subword STT).")
            except Exception as e:
                error(f"Failed to load Whisper model: {e}")
                warn("Falling back to Vosk...")
                self.stt_engine = "vosk"

        # ── Load Vosk as fallback ───────────────────────────────────────────
        if self.stt_engine == "vosk" or (self.stt_engine == "whisper" and not self._whisper_model):
            self.stt_engine = "vosk"
            if not VOSK_AVAILABLE:
                warn("Neither Whisper nor Vosk available for STT.")
                return
            backend = "pyaudio" if self._use_pyaudio else "sounddevice"
            info(f"Loading Vosk model (backend: {backend})...")
            try:
                if self.vosk_model_path and os.path.isdir(self.vosk_model_path):
                    self._vosk_model = vosk.Model(self.vosk_model_path)
                else:
                    self._vosk_model = vosk.Model(lang="en-us")
                success("Vosk model ready (dictionary-based, CPU).")
            except Exception as e:
                error(f"Failed to load Vosk model: {e}")

    def unload_models(self):
        """Unload STT model weights from RAM/GPU completely."""
        with self._mic_lock:
            info("Unloading STT model weights from memory...")
            self._whisper_model = None
            self._vosk_model = None
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            success("STT models completely unloaded (0 MB RAM/GPU consumption).")

    def reload_models(self):
        """Reload STT model weights into memory."""
        with self._mic_lock:
            info("Reloading STT model weights into memory...")
            self._init_models()

    def set_context_bias(self, prompt: str) -> None:
        """Set contextual bias string for Whisper's initial_prompt."""
        self._context_bias = prompt


    # ── Calibration ────────────────────────────────────────────────────────

    def calibrate(self) -> None:
        """Calibrate microphone for ambient noise."""
        info("Calibrating microphone for ambient noise...")
        if self._use_pyaudio and self.recognizer and self.mic:
            with self.mic as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=AMBIENT_DURATION)
            self._calibrated = True
        elif SOUNDDEVICE_AVAILABLE:
            self._calibrate_sounddevice()
        success("Microphone calibrated.")

    def _calibrate_sounddevice(self) -> None:
        """Measure ambient noise level to set dynamic VAD threshold."""
        RATE, CHUNK = 16000, 512
        n_chunks = int(RATE / CHUNK * AMBIENT_DURATION)
        energies = []
        try:
            with sd.RawInputStream(samplerate=RATE, blocksize=CHUNK,
                                   channels=1, dtype="int16") as stream:
                for _ in range(n_chunks):
                    data, _ = stream.read(CHUNK)
                    energies.append(float(np.abs(
                        np.frombuffer(data, dtype=np.int16)
                    ).mean()))
            ambient = float(np.mean(energies))
            self._sd_energy = max(ambient * 4, 200)
            self._calibrated = True
        except Exception as e:
            warn(f"sounddevice calibration failed: {e}")

    # ── Listening ──────────────────────────────────────────────────────────

    def listen_once(self) -> Optional[str]:
        """Listen for one utterance and return transcribed text or None."""
        if self._use_pyaudio:
            return self._listen_pyaudio(LISTEN_TIMEOUT, PHRASE_LIMIT)
        else:
            return self._listen_sounddevice(LISTEN_TIMEOUT, PHRASE_LIMIT)

    def listen_once_timeout(self, timeout: float = 5.0) -> Optional[str]:
        """Listen with custom timeout."""
        if self._use_pyaudio:
            return self._listen_pyaudio(timeout, PHRASE_LIMIT)
        else:
            return self._listen_sounddevice(timeout, PHRASE_LIMIT)

    def _listen_pyaudio(self, timeout: float, phrase_limit: float) -> Optional[str]:
        """Capture audio via PyAudio + SpeechRecognition, then transcribe."""
        if not self.recognizer or not self.mic:
            return None
        with self._mic_lock:
            try:
                with self.mic as source:
                    audio = self.recognizer.listen(
                        source, timeout=timeout, phrase_time_limit=phrase_limit
                    )
                return self._transcribe(audio)
            except sr.WaitTimeoutError:
                return None
            except Exception as e:
                warn(f"PyAudio listen error: {e}")
                return None

    def _listen_sounddevice(self, timeout: float, phrase_limit: float,
                            burst_mode: bool = False) -> Optional[str]:
        """Capture audio via sounddevice with energy-based VAD, then transcribe."""
        if not SOUNDDEVICE_AVAILABLE:
            return None
        RATE, CHUNK = 16000, 512
        max_chunks = int(RATE / CHUNK * timeout)
        phrase_chunks = int(RATE / CHUNK * phrase_limit)
        silence_limit = int(RATE / CHUNK * 1.2)

        frames = []
        recording = False
        silence_count = 0

        with self._mic_lock:
            try:
                with sd.RawInputStream(samplerate=RATE, blocksize=CHUNK,
                                       channels=1, dtype="int16") as stream:
                    for i in range(max_chunks):
                        data, _ = stream.read(CHUNK)
                        chunk_bytes = bytes(data)
                        energy = float(np.abs(
                            np.frombuffer(chunk_bytes, dtype=np.int16)
                        ).mean())

                        if energy > self._sd_energy:
                            if not recording:
                                recording = True
                            silence_count = 0
                            frames.append(chunk_bytes)
                        elif recording:
                            silence_count += 1
                            frames.append(chunk_bytes)
                            if silence_count > silence_limit:
                                break
                        # Enforce phrase limit
                        if recording and len(frames) >= phrase_chunks:
                            break
            except Exception as e:
                warn(f"sounddevice listen error: {e}")
                return None

        if not frames:
            return None

        raw_bytes = b"".join(frames)
        # Build an AudioData-like object for transcription
        return self._transcribe_raw(raw_bytes, RATE)

    # ── Transcription ──────────────────────────────────────────────────────

    def _transcribe(self, audio) -> Optional[str]:
        """Transcribe a SpeechRecognition AudioData object."""
        if self._whisper_model:
            raw = audio.get_raw_data(convert_rate=16000, convert_width=2)
            return self._transcribe_whisper(raw)
        elif self._vosk_model:
            raw = audio.get_raw_data(convert_rate=16000, convert_width=2)
            return self._transcribe_vosk(raw)
        return None

    def _transcribe_raw(self, raw_bytes: bytes, sample_rate: int = 16000) -> Optional[str]:
        """Transcribe raw PCM 16-bit mono bytes."""
        if self._whisper_model:
            return self._transcribe_whisper(raw_bytes)
        elif self._vosk_model:
            return self._transcribe_vosk(raw_bytes)
        return None

    def _transcribe_whisper(self, raw_pcm: bytes) -> Optional[str]:
        """Transcribe raw PCM audio using faster-whisper BPE model."""
        if not self._whisper_model:
            return None

        # Write PCM to temporary WAV file (faster-whisper needs file/array)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
            with wave.open(f, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(raw_pcm)

        try:
            segments, _info = self._whisper_model.transcribe(
                tmp_path,
                language="en",
                initial_prompt=self._context_bias,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=200),
                word_timestamps=True,
                beam_size=5,
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
        except Exception as e:
            warn(f"Whisper transcription error: {e}")
            text = ""
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        if not text:
            return None

        # Normalize
        normalized = normalize_utterance(text)
        if normalized != text:
            info(f"Phonetic normalized: '{text}' → '{normalized}'")
        return normalized

    def _transcribe_vosk(self, raw_pcm: bytes) -> Optional[str]:
        """Transcribe raw PCM audio using Vosk."""
        if not self._vosk_model:
            return None

        rec = vosk.KaldiRecognizer(self._vosk_model, 16000)
        rec.AcceptWaveform(raw_pcm)
        result = json.loads(rec.FinalResult())
        text = result.get("text", "").strip()

        if not text:
            return None

        normalized = normalize_utterance(text)
        if normalized != text:
            info(f"Phonetic normalized: '{text}' → '{normalized}'")
        return normalized

    # ── Stop listener ──────────────────────────────────────────────────────

    def listen_for_stop(self, stop_words: tuple = ("stop",), done_event=None) -> bool:
        """Listen for stop words while TTS is playing. Returns True if stop detected."""
        while done_event is None or not done_event.is_set():
            text = self.listen_once_timeout(timeout=1.5)
            if text and any(w in text.lower().split() for w in stop_words):
                return True
            if done_event and done_event.is_set():
                break
        return False

    def contains_wake_word(self, text: str, wake_word: str) -> bool:
        """Check if text contains the wake word (with phonetic fuzzy matching)."""
        if not text or not wake_word:
            return False
        text_lower = text.lower()
        wake_lower = wake_word.lower()

        # Direct match
        if wake_lower in text_lower.split():
            return True

        # Fuzzy match for close phonetics
        import difflib
        for word in text_lower.split():
            ratio = difflib.SequenceMatcher(None, word, wake_lower).ratio()
            if ratio >= 0.75:
                return True
        return False


# ─── STT Engine selection ─────────────────────────────────────────────────────

def select_stt_engine(args, config: dict) -> Tuple[str, str, Optional[str]]:
    """Interactive STT engine picker. Returns (stt_engine, whisper_model, vosk_model_path)."""

    # Restore from config if not reconfiguring
    if not getattr(args, "reconfigure", False):
        if config.get("stt_engine") and config.get("whisper_model"):
            stt = config["stt_engine"]
            wm = config["whisper_model"]
            vp = config.get("vosk_model_path")
            info(f"Loaded STT from .config: {stt} ({wm})")
            return stt, wm, vp

    # Check if Whisper is available
    if not FASTER_WHISPER_AVAILABLE:
        warn("faster-whisper not installed. Using Vosk fallback.")
        vosk_path = _resolve_vosk_path(getattr(args, "vosk_model", None))
        return "vosk", "", vosk_path

    print(f"\n{BOLD}{CYAN}Select Speech-to-Text Engine:{RESET}")
    print(c("─" * 70, DIM))

    # List all Whisper models
    for i, m in enumerate(WHISPER_MODELS, 1):
        tag = c(f"[{i:2d}]", CYAN)
        default = c(" ◀ recommended", GREEN) if m["name"] == "small.en" else ""
        print(f"  {tag} Whisper {m['desc']}{default}")

    # Add Vosk fallback
    vosk_idx = len(WHISPER_MODELS) + 1
    print(f"  {c(f'[{vosk_idx:2d}]', CYAN)} {VOSK_FALLBACK_MODEL['desc']}")
    print(c("─" * 70, DIM))
    default_idx = 6  # small.en
    print(f"  {c('[Enter]', DIM)} use default ({WHISPER_MODELS[default_idx - 1]['name']})")
    print()

    while True:
        try:
            raw = input(f"{BOLD}Your choice (1-{vosk_idx}): {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return "whisper", "small.en", None

        if not raw:
            return "whisper", "small.en", None
        if raw.isdigit():
            choice = int(raw)
            if 1 <= choice <= len(WHISPER_MODELS):
                model = WHISPER_MODELS[choice - 1]
                success(f"Selected: Whisper {model['name']}")
                return "whisper", model["name"], None
            elif choice == vosk_idx:
                vosk_path = _resolve_vosk_path(getattr(args, "vosk_model", None))
                success(f"Selected: Vosk (fallback)")
                return "vosk", "", vosk_path
        print(c(f"  Please enter a number between 1 and {vosk_idx}", DIM))


def _resolve_vosk_path(explicit_path: Optional[str]) -> Optional[str]:
    """Find or download the Vosk fallback model."""
    if explicit_path and os.path.isdir(explicit_path):
        return explicit_path

    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "vosk-models")
    model_name = VOSK_FALLBACK_MODEL["name"]
    model_path = os.path.join(models_dir, model_name)

    if os.path.isdir(model_path):
        return model_path

    # Check if any Vosk model exists
    if os.path.isdir(models_dir):
        for d in os.listdir(models_dir):
            p = os.path.join(models_dir, d)
            if os.path.isdir(p) and os.path.isfile(os.path.join(p, "am", "final.mdl")):
                return p

    # Download
    info(f"Downloading Vosk fallback model: {model_name}...")
    download_and_extract_vosk(VOSK_FALLBACK_MODEL, models_dir)
    return model_path


def download_and_extract_vosk(model_info: dict, dest_dir: str) -> None:
    """Download and extract a Vosk model."""
    os.makedirs(dest_dir, exist_ok=True)
    name = model_info["name"]
    url = model_info["url"]
    zip_path = os.path.join(dest_dir, f"{name}.zip")

    print(f"\n{CYAN}Downloading {name}...{RESET}")
    try:
        def reporthook(count, block_size, total_size):
            if total_size > 0:
                percent = int(count * block_size * 100 / total_size)
                print(f"\r  Progress: {percent}% ", end="", flush=True)
        urllib.request.urlretrieve(url, zip_path, reporthook)
        print(f"\n{CYAN}Extracting...{RESET}")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(dest_dir)
        os.remove(zip_path)
        success(f"Successfully installed {name}!")
    except Exception as e:
        error(f"Failed to download {name}: {e}")
        if os.path.exists(zip_path):
            os.remove(zip_path)
