#!/usr/bin/env python3
"""
🎙️ Local AI Voice Assistant
----------------------------
Wake word: "hey"
STT:       Vosk (100% on-device, no internet required)
LLM:       Ollama (any local model, runs locally)
TTS:       pyttsx3 (offline) or edge-tts (higher quality, needs internet)

Usage:
    python assistant.py
    python assistant.py --model llama3.1:8b
    python assistant.py --vosk-model /path/to/vosk-model-en
    python assistant.py --model mistral --tts edge
    python assistant.py --list-voices
"""

import argparse
import json
import os
import platform as _platform
import queue
import subprocess
import sys
import tempfile
import threading
import time
from typing import Optional

# ─── Colour helpers ───────────────────────────────────────────────────────────
RESET   = "\033[0m"
BOLD    = "\033[1m"
CYAN    = "\033[96m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
MAGENTA = "\033[95m"
DIM     = "\033[2m"

def c(text, colour): return f"{colour}{text}{RESET}"
def info(msg):    print(c(f"[•] {msg}", CYAN))
def success(msg): print(c(f"[✓] {msg}", GREEN))
def warn(msg):    print(c(f"[!] {msg}", YELLOW))
def error(msg):   print(c(f"[✗] {msg}", RED), file=sys.stderr)
def speak_label(msg): print(c(f"[🔊] {msg}", MAGENTA))
def user_label(msg):  print(c(f"[🎤] {msg}", GREEN))
def ai_label(msg):    print(c(f"[🤖] {msg}", CYAN))

# ─── Imports (with friendly error messages) ───────────────────────────────────
def require(package, install_hint):
    try:
        return __import__(package)
    except ImportError:
        error(f"Missing package '{package}'. Install with: {install_hint}")
        sys.exit(1)

speech_recognition = require("speech_recognition", "pip install SpeechRecognition")
sr = speech_recognition

try:
    import vosk
    vosk.SetLogLevel(-1)  # suppress verbose Vosk output
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False
    error("Vosk not found. Install with: pip install vosk")
    error("Then download a model: https://alphacephei.com/vosk/models")
    sys.exit(1)

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    warn("pyttsx3 not found. Install with: pip install pyttsx3")

try:
    import edge_tts
    import asyncio
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False

try:
    import ollama as ollama_client
    OLLAMA_LIB_AVAILABLE = True
except ImportError:
    OLLAMA_LIB_AVAILABLE = False
    warn("ollama Python library not found. Using HTTP API fallback.")
    import urllib.request, json as _json

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

try:
    from piper.voice import PiperVoice as _PiperVoiceClass
    PIPER_AVAILABLE = True
except ImportError:
    PIPER_AVAILABLE = False

# ── Audio input library (PyAudio preferred, sounddevice as fallback) ───────────
try:
    import pyaudio as _pyaudio_check
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

try:
    import sounddevice as _sd_check
    import numpy as _np_check
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False

if not PYAUDIO_AVAILABLE and not SOUNDDEVICE_AVAILABLE:
    error("No audio input library found. Install one of:")
    error("  pip install pyaudio      (also needs portaudio system package)")
    error("  pip install sounddevice numpy  (easier, no system deps on most platforms)")
    sys.exit(1)

if not PYAUDIO_AVAILABLE:
    warn("PyAudio not found — using sounddevice for microphone input.")
    import sounddevice as sd
    import numpy as np
elif SOUNDDEVICE_AVAILABLE:
    import sounddevice as sd
    import numpy as np

# ─── Configuration ─────────────────────────────────────────────────────────────
DEFAULT_MODEL      = "llama3.1:8b"
DEFAULT_WAKE_WORD  = "hey"
DEFAULT_TTS        = "piper"   # piper → pyttsx3 → system-native → edge-tts
PIPER_VOICE_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "piper-voices")
LISTEN_TIMEOUT     = 8          # seconds to wait for speech after wake word
PHRASE_LIMIT       = 15         # max seconds per utterance
AMBIENT_DURATION   = 1.0        # seconds to calibrate microphone noise
CONVERSATION_HIST  = 10         # how many turns to keep in context

SYSTEM_PROMPT = """You are a helpful, conversational AI voice assistant.
Keep your responses concise and natural for speech — aim for 1-3 sentences unless more detail is clearly needed.
Do not use markdown formatting, bullet points, or special characters in responses.
Speak naturally as if in a conversation."""

# ─── TTS Engine ────────────────────────────────────────────────────────────────
class TTSEngine:
    def __init__(self, mode: str = "pyttsx3", voice_id: Optional[str] = None,
                 rate: int = 175, piper_voice_path: Optional[str] = None):
        self.mode         = mode
        self.voice_id     = voice_id
        self.rate         = rate
        self._engine      = None
        self._piper_voice = None
        self._sys_proc: Optional[subprocess.Popen] = None
        self._lock        = threading.Lock()
        self._stop_event  = threading.Event()
        self._speaking    = False

        # ── Piper (neural, offline, best quality) ───────────────────────────────
        if mode == "piper":
            if not PIPER_AVAILABLE:
                warn("piper-tts not found — falling back to pyttsx3.  pip install piper-tts")
                self.mode = "pyttsx3"
            elif piper_voice_path and os.path.isfile(piper_voice_path):
                try:
                    info("Loading Piper TTS neural voice model...")
                    cfg = piper_voice_path + ".json"
                    self._piper_voice = _PiperVoiceClass.load(
                        piper_voice_path,
                        config_path=cfg if os.path.isfile(cfg) else None,
                        use_cuda=False,
                    )
                    success("Piper TTS ready (neural, fully offline).")
                except Exception as e:
                    warn(f"Piper voice load failed ({e}) — falling back to pyttsx3.")
                    self.mode = "pyttsx3"
            else:
                warn("No Piper voice model found — falling back to pyttsx3.")
                self.mode = "pyttsx3"

        # ── pyttsx3 (offline, cross-platform) ─────────────────────────────────
        if mode == "pyttsx3" or self.mode == "pyttsx3":
            self.mode = "pyttsx3"
            if not PYTTSX3_AVAILABLE:
                warn("pyttsx3 unavailable — falling back to system-native TTS.")
                self.mode = "system"
            else:
                try:
                    self._engine = pyttsx3.init()
                    self._engine.setProperty("rate", rate)
                    if voice_id:
                        self._engine.setProperty("voice", voice_id)
                except Exception as e:
                    warn(f"pyttsx3 init failed ({e}) — falling back to system-native TTS.")
                    self.mode = "system"

        # ── System-native / edge (remaining modes unchanged) ──────────────────
        if mode == "system" or self.mode == "system":
            self.mode = "system"
            _os = _platform.system()
            if _os == "Darwin":
                info("TTS: using macOS built-in 'say' command.")
            elif _os == "Windows":
                info("TTS: using Windows SAPI via PowerShell.")
            else:  # Linux / other
                _espeak = subprocess.run(["which", "espeak-ng"], capture_output=True).returncode == 0 \
                          or subprocess.run(["which", "espeak"], capture_output=True).returncode == 0
                if not _espeak:
                    warn("espeak / espeak-ng not found. Install: sudo apt install espeak-ng")
                    warn("Falling back to edge-tts for TTS.")
                    self.mode = "edge"
                else:
                    info("TTS: using espeak-ng (Linux native).")

        elif mode == "edge" or self.mode == "edge":
            if not EDGE_TTS_AVAILABLE:
                warn("edge-tts unavailable. Install: pip install edge-tts pygame")
                self.mode = "none"

    def list_voices(self):
        if self.mode == "pyttsx3" and self._engine:
            voices = self._engine.getProperty("voices")
            print(f"\n{BOLD}Available pyttsx3 voices:{RESET}")
            for v in voices:
                print(f"  ID:   {c(v.id, CYAN)}")
                print(f"  Name: {v.name}")
                print(f"  Lang: {v.languages}\n")
        elif self.mode == "edge":
            async def _list():
                voices = await edge_tts.list_voices()
                print(f"\n{BOLD}Available edge-tts voices (English subset):{RESET}")
                for v in voices:
                    if "en-" in v["Locale"].lower():
                        print(f"  {c(v['ShortName'], CYAN)} — {v['FriendlyName']}")
            asyncio.run(_list())

    # ── Public API ──────────────────────────────────────────────────────────────

    @property
    def is_speaking(self) -> bool:
        return self._speaking

    def stop(self):
        """Interrupt any currently playing speech immediately."""
        self._stop_event.set()
        if self.mode == "pyttsx3" and self._engine:
            try:
                self._engine.stop()
            except Exception:
                pass
        elif self.mode == "system":
            try:
                if self._sys_proc and self._sys_proc.poll() is None:
                    self._sys_proc.terminate()
            except Exception:
                pass
        elif self.mode in ("piper", "edge") and PYGAME_AVAILABLE:
            try:
                pygame.mixer.music.stop()
            except Exception:
                pass

    def speak(self, text: str) -> bool:
        """Speak text. Returns True if completed fully, False if interrupted."""
        if not text.strip():
            return True
        self._stop_event.clear()
        self._speaking = True
        speak_label(text)
        try:
            if self.mode == "piper" and self._piper_voice:
                return self._speak_piper(text)
            elif self.mode == "pyttsx3" and self._engine:
                return self._speak_pyttsx3(text)
            elif self.mode == "system":
                return self._speak_system(text)
            elif self.mode == "edge":
                return asyncio.run(self._speak_edge(text))
            else:
                print(c(f"  [TTS disabled] Would say: {text}", DIM))
                return True
        finally:
            self._speaking = False

    # ── Piper neural TTS ─────────────────────────────────────────────────────

    def _speak_piper(self, text: str) -> bool:
        """Synthesize with Piper (neural) and play via pygame (interruptible)."""
        import wave, io
        buf = io.BytesIO()
        try:
            with wave.open(buf, 'w') as wf:
                self._piper_voice.synthesize_wav(text, wf)
        except Exception as e:
            warn(f"Piper synthesis error: {e}")
            return True

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            f.write(buf.getvalue())
            tmp_path = f.name
        try:
            return self._play_audio_interruptible(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    # ── pyttsx3 interruptible speak ─────────────────────────────────────────────

    def _speak_pyttsx3(self, text: str) -> bool:
        """Run pyttsx3 in a thread; poll stop_event to interrupt it."""
        done_event  = threading.Event()
        interrupted = [False]

        def _worker():
            try:
                with self._lock:
                    self._engine.say(text)
                    self._engine.runAndWait()
            except Exception:
                pass
            finally:
                done_event.set()

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

        # Poll until speech finishes or stop is requested
        while not done_event.wait(timeout=0.05):
            if self._stop_event.is_set():
                interrupted[0] = True
                try:
                    self._engine.stop()
                except Exception:
                    pass
                break

        done_event.wait(timeout=1.0)  # let thread clean up
        return not interrupted[0]

    # ── System-native TTS (macOS / Windows / Linux) ─────────────────────────────

    def _speak_system(self, text: str) -> bool:
        """Use the OS built-in TTS engine (fully offline, no deps)."""
        _os = _platform.system()
        # Escape text for safe shell passing
        safe = text.replace('"', '""').replace("'", "'\"'\"'")

        try:
            if _os == "Darwin":                          # macOS — 'say' command
                cmd = ["say", text]
            elif _os == "Windows":                       # Windows — PowerShell SAPI
                ps = (
                    f'Add-Type -AssemblyName System.Speech; '
                    f'$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; '
                    f'$s.Rate = {max(-10, min(10, (self.rate - 175) // 25))}; '
                    f'$s.Speak("{text.replace(chr(34), chr(39))}")'
                )
                cmd = ["powershell", "-NoProfile", "-Command", ps]
            else:                                        # Linux — espeak-ng / espeak
                espeak = "espeak-ng" if subprocess.run(
                    ["which", "espeak-ng"], capture_output=True
                ).returncode == 0 else "espeak"
                cmd = [espeak, "-s", str(self.rate), text]

            self._sys_proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            # Poll for completion or stop signal
            while self._sys_proc.poll() is None:
                if self._stop_event.is_set():
                    self._sys_proc.terminate()
                    return False
                time.sleep(0.05)
            return True

        except FileNotFoundError as e:
            warn(f"System TTS command not found: {e}")
            print(c(f"  [TTS fallback] {text}", DIM))
            return True
        except Exception as e:
            warn(f"System TTS error: {e}")
            return True

    # ── edge-tts interruptible speak ────────────────────────────────────────────

    async def _speak_edge(self, text: str) -> bool:
        voice = self.voice_id or "en-US-GuyNeural"
        tts   = edge_tts.Communicate(text, voice)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp_path = f.name
        try:
            await tts.save(tmp_path)
            return self._play_audio_interruptible(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _play_audio_interruptible(self, path: str) -> bool:
        """Play audio file; returns False if stopped early."""
        if PYGAME_AVAILABLE:
            pygame.mixer.init()
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                if self._stop_event.is_set():
                    pygame.mixer.music.stop()
                    return False
                time.sleep(0.05)
            return True
        else:
            # Fallback: system player (not interruptible)
            _os = _platform.system()
            if _os == "Darwin":
                cmd = f"afplay '{path}'"
            elif _os == "Windows":
                cmd = f'powershell -NoProfile -Command "(New-Object System.Media.SoundPlayer \'{path}\').PlaySync()"'
            else:
                if path.endswith(".wav"):
                    cmd = f"aplay -q '{path}' 2>/dev/null || paplay '{path}' 2>/dev/null"
                else:
                    cmd = f"mpg123 -q '{path}' 2>/dev/null"
                cmd += f" || ffplay -nodisp -autoexit '{path}' 2>/dev/null"
            os.system(cmd)
            return True


# ─── Speech Recognition ────────────────────────────────────────────────────────
class SpeechListener:
    def __init__(self, timeout: int = LISTEN_TIMEOUT, phrase_limit: int = PHRASE_LIMIT,
                 vosk_model_path: Optional[str] = None):
        self.recognizer   = sr.Recognizer()
        self.timeout      = timeout
        self.phrase_limit = phrase_limit
        self._calibrated  = False
        self._mic_lock    = threading.Lock()  # only one thread uses the mic at a time
        self._use_pyaudio = PYAUDIO_AVAILABLE  # which audio backend
        self._sd_energy   = 300                # sounddevice VAD threshold (set during calibration)

        if self._use_pyaudio:
            self.mic = sr.Microphone()
        else:
            self.mic = None  # sounddevice path — no sr.Microphone needed

        # ── Load Vosk model ──────────────────────────────────────────────────
        info(f"Loading Vosk speech recognition model (on-device, backend: {'pyaudio' if self._use_pyaudio else 'sounddevice'})...")
        try:
            if vosk_model_path and os.path.isdir(vosk_model_path):
                self._vosk_model = vosk.Model(vosk_model_path)
            else:
                # Auto-download the small English model (~50 MB) on first run
                self._vosk_model = vosk.Model(lang="en-us")
            success("Vosk model ready (fully offline).")
        except Exception as e:
            error(f"Failed to load Vosk model: {e}")
            error("Download a model from https://alphacephei.com/vosk/models")
            error("Extract it and pass the path with:  --vosk-model /path/to/model")
            sys.exit(1)

    def calibrate(self):
        info("Calibrating microphone for ambient noise...")
        if self._use_pyaudio:
            with self.mic as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=AMBIENT_DURATION)
            self._calibrated = True
        else:
            self._calibrate_sounddevice()
        success("Microphone calibrated.")

    def _calibrate_sounddevice(self):
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
            self._sd_energy = max(ambient * 4, 200)  # 4× ambient, minimum 200
            self._calibrated = True
        except Exception as e:
            warn(f"sounddevice calibration failed: {e}")

    def listen_once(self) -> Optional[str]:
        """Listen for one utterance and return transcribed text or None."""
        if self._use_pyaudio:
            return self._listen_pyaudio(self.timeout, self.phrase_limit)
        else:
            return self._listen_sounddevice(self.timeout, self.phrase_limit)

    def _listen_pyaudio(self, timeout, phrase_limit) -> Optional[str]:
        """Capture audio via PyAudio + SpeechRecognition."""
        with self._mic_lock:
            with self.mic as source:
                if not self._calibrated:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.3)
                try:
                    audio = self.recognizer.listen(
                        source,
                        timeout=timeout,
                        phrase_time_limit=phrase_limit
                    )
                except sr.WaitTimeoutError:
                    return None
        return self._transcribe(audio)

    def _listen_sounddevice(self, timeout, phrase_limit,
                             burst_mode=False) -> Optional[str]:
        """Capture audio via sounddevice with energy-based VAD."""
        RATE  = 16000
        CHUNK = 512
        thresh = self._sd_energy
        max_phrase = int(RATE / CHUNK * phrase_limit)
        timeout_chunks = int(RATE / CHUNK * timeout)
        silence_limit  = int(RATE / CHUNK * 1.2)  # 1.2 s silence = end of phrase

        frames: list = []
        speech_started = False
        silence_count  = 0
        waited         = 0

        try:
            with sd.RawInputStream(samplerate=RATE, blocksize=CHUNK,
                                   channels=1, dtype="int16") as stream:
                while True:
                    data, _ = stream.read(CHUNK)
                    energy  = float(np.abs(
                        np.frombuffer(data, dtype=np.int16)
                    ).mean())

                    if not speech_started:
                        if energy > thresh:
                            speech_started = True
                            frames.append(bytes(data))
                        else:
                            waited += 1
                            if waited > timeout_chunks:
                                return None
                    else:
                        frames.append(bytes(data))
                        if energy < thresh:
                            silence_count += 1
                            if silence_count > silence_limit:
                                break
                        else:
                            silence_count = 0
                        if len(frames) > max_phrase:
                            break
        except Exception as e:
            warn(f"sounddevice capture error: {e}")
            return None

        audio = sr.AudioData(b"".join(frames), RATE, 2)
        return self._transcribe(audio)

    def listen_for_stop(self, stop_words: tuple = ("stop",), done_event: threading.Event = None) -> bool:
        """
        Continuously listen in short bursts until one of stop_words is heard
        or done_event is set. Returns True if a stop word was detected.
        """
        while done_event is None or not done_event.is_set():
            if self._use_pyaudio:
                # PyAudio path — use lock to avoid mic conflict
                if not self._mic_lock.acquire(blocking=True, timeout=0.2):
                    continue
                try:
                    with self.mic as source:
                        try:
                            audio = self.recognizer.listen(
                                source, timeout=1.5, phrase_time_limit=2.0
                            )
                        except sr.WaitTimeoutError:
                            continue
                finally:
                    self._mic_lock.release()
                text = self._transcribe(audio)
            else:
                # sounddevice path — no lock needed (separate stream)
                text = self._listen_sounddevice(
                    timeout=1.5, phrase_limit=2.0, burst_mode=True
                )

            if text and any(w in text.lower().split() for w in stop_words):
                return True

        return False

    def _transcribe(self, audio, recognizer=None) -> Optional[str]:
        """Transcribe audio to text using Vosk (fully offline, no internet)."""
        try:
            # Convert captured audio to 16kHz mono 16-bit PCM for Vosk
            raw = audio.get_raw_data(convert_rate=16000, convert_width=2)
            rec = vosk.KaldiRecognizer(self._vosk_model, 16000)
            rec.AcceptWaveform(raw)
            result = json.loads(rec.FinalResult())
            text = result.get("text", "").strip()
            return text if text else None
        except Exception as e:
            warn(f"Vosk transcription error: {e}")
            return None

    def contains_wake_word(self, text: str, wake_word: str) -> bool:
        return wake_word.lower() in text.lower().split()


# ─── Ollama Client ─────────────────────────────────────────────────────────────
class OllamaLLM:
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model   = model
        self.history = []  # list of {"role": ..., "content": ...}

    def reset(self):
        self.history.clear()

    def _trim_history(self):
        if len(self.history) > CONVERSATION_HIST * 2:
            self.history = self.history[-(CONVERSATION_HIST * 2):]

    def chat(self, user_input: str) -> str:
        self.history.append({"role": "user", "content": user_input})
        self._trim_history()

        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.history

        if OLLAMA_LIB_AVAILABLE:
            return self._chat_lib(messages)
        else:
            return self._chat_http(messages)

    def _chat_lib(self, messages: list) -> str:
        try:
            response = ollama_client.chat(
                model=self.model,
                messages=messages,
                stream=False
            )
            reply = response["message"]["content"].strip()
            self.history.append({"role": "assistant", "content": reply})
            return reply
        except Exception as e:
            error(f"Ollama error: {e}")
            self.history.pop()  # rollback
            return "Sorry, I had trouble generating a response."

    def _chat_http(self, messages: list) -> str:
        payload = _json.dumps({
            "model":    self.model,
            "messages": messages,
            "stream":   False
        }).encode()
        req = urllib.request.Request(
            "http://localhost:11434/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data  = _json.loads(resp.read())
                reply = data["message"]["content"].strip()
                self.history.append({"role": "assistant", "content": reply})
                return reply
        except Exception as e:
            error(f"Ollama HTTP error: {e}")
            self.history.pop()
            return "Sorry, I could not reach the Ollama server."

    def check_available(self) -> bool:
        """Return True if the Ollama server is reachable."""
        if OLLAMA_LIB_AVAILABLE:
            try:
                ollama_client.list()
                return True
            except Exception:
                return False
        else:
            try:
                urllib.request.urlopen("http://localhost:11434", timeout=3)
                return True
            except Exception:
                return False


# ─── Main Assistant Loop ───────────────────────────────────────────────────────
class VoiceAssistant:
    def __init__(self, model: str, tts_mode: str, voice_id: Optional[str], tts_rate: int,
                 wake_word: str, vosk_model_path: Optional[str] = None,
                 piper_voice_path: Optional[str] = None):
        self.wake_word = wake_word.lower()
        self.llm       = OllamaLLM(model=model)
        self.tts       = TTSEngine(mode=tts_mode, voice_id=voice_id, rate=tts_rate,
                                   piper_voice_path=piper_voice_path)
        self.listener  = SpeechListener(vosk_model_path=vosk_model_path)
        self._running  = False

    def _print_banner(self):
        banner = f"""
{BOLD}{CYAN}╔══════════════════════════════════════════════╗
║       🎙️  Local AI Voice Assistant           ║
╚══════════════════════════════════════════════╝{RESET}
  Model    : {c(self.llm.model, GREEN)}
  Wake word: {c(f'"{self.wake_word}"', YELLOW)}
  TTS      : {c(self.tts.mode, MAGENTA)}
  Commands : {c('say "reset" to clear history', DIM)}
             {c('say "stop" to interrupt AI speech', DIM)}
             {c('say "quit" or press Ctrl+C to exit', DIM)}
"""
        print(banner)

    def run(self):
        self._print_banner()

        # Check Ollama
        info(f"Connecting to Ollama (model: {self.llm.model})...")
        if not self.llm.check_available():
            error("Ollama server not reachable at http://localhost:11434")
            error("Start it with: ollama serve")
            sys.exit(1)
        success(f"Ollama ready with model '{self.llm.model}'")

        # Calibrate mic
        self.listener.calibrate()

        self.tts.speak(f"Hello! Say {self.wake_word} to wake me up.")

        print(f"\n{BOLD}Listening for wake word: \"{self.wake_word}\" ...{RESET}\n")

        self._running = True
        try:
            while self._running:
                self._wait_for_wake_word()
        except KeyboardInterrupt:
            print(f"\n{DIM}Interrupted by user.{RESET}")
        finally:
            success("Goodbye!")
            self.tts.speak("Goodbye!")

    def _wait_for_wake_word(self):
        """Continuously listen until wake word is detected."""
        text = self.listener.listen_once()
        if text is None:
            return

        print(f"{DIM}heard: {text}{RESET}")

        if self.listener.contains_wake_word(text, self.wake_word):
            # Extract any query that came right after "hey <...>"
            words = text.lower().split()
            idx   = next((i for i, w in enumerate(words) if w == self.wake_word), -1)
            inline_query = " ".join(words[idx + 1:]).strip() if idx >= 0 else ""

            print(f"\n{BOLD}{GREEN}⚡ Wake word detected!{RESET}")
            self.tts.speak("Yes?")

            self._handle_query(inline_query)

    def _handle_query(self, prefill: str = ""):
        """Capture command (with optional prefill) and respond."""
        if prefill:
            user_input = prefill
            user_label(user_input)
        else:
            info("Listening for your question...")
            user_input = self.listener.listen_once()
            if not user_input:
                warn("Didn't catch that. Please try again.")
                self.tts.speak("I didn't catch that. Please try again.")
                return
            user_label(user_input)

        # Special commands
        cmd = user_input.lower().strip()
        if cmd in ("quit", "exit", "bye", "goodbye"):
            self._running = False
            return
        if cmd in ("reset", "clear history", "start over"):
            self.llm.reset()
            success("Conversation history cleared.")
            self.tts.speak("Okay, starting fresh!")
            return

        # Query LLM
        info("Thinking...")
        response = self.llm.chat(user_input)
        ai_label(response)

        # Speak response (interruptible with "stop")
        self._speak_with_stop_listener(response)
        print(f"\n{BOLD}Listening for wake word: \"{self.wake_word}\" ...{RESET}\n")

    def _speak_with_stop_listener(self, text: str):
        """
        Speak text while a background thread listens for the word "stop".
        If heard, TTS is interrupted immediately.
        """
        speech_done = threading.Event()
        stopped_by_user = [False]

        def _stop_listener():
            detected = self.listener.listen_for_stop(
                stop_words=("stop",),
                done_event=speech_done
            )
            if detected and self.tts.is_speaking:
                stopped_by_user[0] = True
                warn("Stopping speech...")
                self.tts.stop()

        # Only start the stop-listener if mic locking is possible
        listener_thread = threading.Thread(target=_stop_listener, daemon=True)
        listener_thread.start()

        self.tts.speak(text)

        # Signal the listener thread that speech is over
        speech_done.set()
        listener_thread.join(timeout=2.0)

        if stopped_by_user[0]:
            print(c("[✋] Response stopped by user.", YELLOW))


# ─── Ollama model selector ─────────────────────────────────────────────────────
def list_ollama_models() -> list:
    """Return list of locally installed Ollama model names."""
    try:
        if OLLAMA_LIB_AVAILABLE:
            result = ollama_client.list()
            # Handle both dict and object responses across library versions
            models_raw = result.get("models", []) if isinstance(result, dict) else getattr(result, "models", [])
            return [m["name"] if isinstance(m, dict) else m.model for m in models_raw]
        else:
            import urllib.request as _ur
            with _ur.urlopen("http://localhost:11434/api/tags", timeout=5) as r:
                data = _json.loads(r.read())
                return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def select_ollama_model(explicit_model: Optional[str]) -> str:
    """
    If explicit_model is given, return it directly.
    Otherwise query Ollama for installed models and present a numbered menu.
    Falls back to DEFAULT_MODEL if nothing is available.
    """
    if explicit_model:
        return explicit_model

    models = list_ollama_models()

    if not models:
        warn("No Ollama models found or Ollama not running.")
        warn(f"Defaulting to '{DEFAULT_MODEL}'. Pull it with:  ollama pull {DEFAULT_MODEL}")
        return DEFAULT_MODEL

    if len(models) == 1:
        success(f"Using the only installed model: {c(models[0], GREEN)}")
        return models[0]

    # ── Interactive picker ────────────────────────────────────────────────────
    print(f"\n{BOLD}{CYAN}Select an Ollama model:{RESET}")
    print(c("─" * 44, DIM))
    for i, name in enumerate(models, 1):
        tag = c(f"[{i}]", CYAN)
        default_marker = c(" ◀ default", DIM) if name == DEFAULT_MODEL else ""
        print(f"  {tag} {name}{default_marker}")
    print(c("─" * 44, DIM))
    print(f"  {c('[Enter]', DIM)} use default ({DEFAULT_MODEL if DEFAULT_MODEL in models else models[0]})")
    print()

    fallback = DEFAULT_MODEL if DEFAULT_MODEL in models else models[0]
    while True:
        try:
            raw = input(f"{BOLD}Your choice (1-{len(models)}): {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return fallback
        if not raw:
            return fallback
        if raw.isdigit() and 1 <= int(raw) <= len(models):
            chosen = models[int(raw) - 1]
            success(f"Selected model: {c(chosen, GREEN)}")
            return chosen
        print(c(f"  Please enter a number between 1 and {len(models)}", YELLOW))


# ─── Vosk model selector & downloader ────────────────────────────────────────────

# Bundled local zip (ships alongside the assistant)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_LGRAPH_LOCAL_ZIP = os.path.join(_SCRIPT_DIR, "vosk-model-en-us-0.22-lgraph.zip")

VOSK_STANDARD_MODELS = [
    {
        "name": "vosk-model-small-en-us-0.15",
        "desc": "Small (40 MB) — Fast, good for older / low-power CPUs",
        "url": "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
    },
    {
        "name": "vosk-model-en-us-0.22-lgraph",
        "desc": "Medium (128 MB) — Good accuracy, low RAM, recommended",
        # Prefer the bundled local zip; fall back to remote download
        "localZip": _LGRAPH_LOCAL_ZIP,
        "url": "https://alphacephei.com/vosk/models/vosk-model-en-us-0.22-lgraph.zip"
    },
]

def download_and_extract_vosk(model_info: dict, dest_dir: str):
    """Install a Vosk model. Uses a local zip if available, otherwise downloads."""
    import urllib.request as _ur
    import zipfile

    name = model_info["name"]
    local_zip = model_info.get("localZip", "")
    os.makedirs(dest_dir, exist_ok=True)

    # ── Prefer the bundled local zip ─────────────────────────────────────────
    if local_zip and os.path.isfile(local_zip):
        info(f"Found bundled zip for {name} — installing locally (no download needed).")
        try:
            print(f"{CYAN}Extracting {os.path.basename(local_zip)}...{RESET}")
            with zipfile.ZipFile(local_zip, 'r') as zip_ref:
                zip_ref.extractall(dest_dir)
            success(f"Successfully extracted {name}!")
            return
        except Exception as e:
            warn(f"Local zip extraction failed ({e}) — falling back to download.")

    # ── Network download fallback ─────────────────────────────────────────────
    url = model_info["url"]
    zip_path = os.path.join(dest_dir, f"{name}.zip")
    print(f"\n{CYAN}Downloading {name}...{RESET}")
    try:
        def reporthook(count, block_size, total_size):
            if total_size > 0:
                percent = int(count * block_size * 100 / total_size)
                print(f"\r  Progress: {percent}% ", end="", flush=True)

        _ur.urlretrieve(url, zip_path, reporthook)
        print(f"\n{CYAN}Extracting...{RESET}")

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)

        os.remove(zip_path)
        success(f"Successfully downloaded and extracted {name}!")
    except Exception as e:
        error(f"Failed to download/extract {name}: {e}")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        sys.exit(1)

def select_vosk_model(explicit_path: Optional[str]) -> str:
    """
    If explicit_path is given, use it.
    Otherwise, present a menu of downloaded and standard Vosk models.
    """
    if explicit_path and os.path.exists(explicit_path):
        return explicit_path

    # Check local vosk-models dir and old vosk-model dir
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "vosk-models")
    old_model_dir = os.path.join(base_dir, "vosk-model")
    
    downloaded_models = []
    
    # Check if old single dir exists
    if os.path.isdir(old_model_dir) and os.path.isfile(os.path.join(old_model_dir, "am", "final.mdl")):
        downloaded_models.append({
            "name": "vosk-model (legacy dir)",
            "path": old_model_dir,
            "desc": "Currently installed default model"
        })
        
    # Check models_dir
    if os.path.isdir(models_dir):
        for d in os.listdir(models_dir):
            path = os.path.join(models_dir, d)
            if os.path.isdir(path) and os.path.isfile(os.path.join(path, "am", "final.mdl")):
                downloaded_models.append({
                    "name": d,
                    "path": path,
                    "desc": "Local model"
                })

    # Combine with standard models
    choices = []
    for std in VOSK_STANDARD_MODELS:
        # Check if already extracted / downloaded
        local_match = next((m for m in downloaded_models if m["name"] == std["name"]), None)
        if local_match:
            choices.append({
                "name": std["name"],
                "path": local_match["path"],
                "desc": std["desc"],
                "downloaded": True
            })
            # Remove from downloaded list so we don't duplicate
            downloaded_models = [m for m in downloaded_models if m["name"] != std["name"]]
        else:
            # Check whether a bundled local zip is available (no network needed)
            local_zip = std.get("localZip", "")
            has_local_zip = bool(local_zip and os.path.isfile(local_zip))
            choices.append({
                "name": std["name"],
                "path": os.path.join(models_dir, std["name"]),
                "desc": std["desc"],
                "downloaded": False,
                "localZip": local_zip,
                "hasLocalZip": has_local_zip,
                "url": std.get("url", "")
            })
            
    # Add any remaining downloaded models (custom ones)
    for dm in downloaded_models:
        choices.append({
            "name": dm["name"],
            "path": dm["path"],
            "desc": dm["desc"],
            "downloaded": True
        })

    # ── Interactive picker ────────────────────────────────────────────────────
    print(f"\n{BOLD}{CYAN}Select a Vosk Speech-to-Text model:{RESET}")
    print(c("─" * 60, DIM))
    for i, c_info in enumerate(choices, 1):
        tag = c(f"[{i}]", CYAN)
        if c_info["downloaded"]:
            status = c("[Ready]", GREEN)
        elif c_info.get("hasLocalZip"):
            status = c("[Bundled zip — installs offline]", CYAN)
        else:
            status = c("[Download required]", YELLOW)
        print(f"  {tag} {c_info['desc']}")
        print(f"      {c(c_info['name'], DIM)}  {status}")
    print(c("─" * 60, DIM))
    
    default_idx = 1
    # Try to default to the first downloaded one, else 1
    for i, c_info in enumerate(choices, 1):
        if c_info["downloaded"]:
            default_idx = i
            break
            
    print(f"  {c('[Enter]', DIM)} use default ({default_idx})")
    print()

    while True:
        try:
            raw = input(f"{BOLD}Your choice (1-{len(choices)}): {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            sys.exit(0)
            
        if not raw:
            choice = choices[default_idx - 1]
            break
        if raw.isdigit() and 1 <= int(raw) <= len(choices):
            choice = choices[int(raw) - 1]
            break
        print(c(f"  Please enter a number between 1 and {len(choices)}", YELLOW))
        
    if not choice["downloaded"]:
        download_and_extract_vosk(choice, models_dir)
        
    success(f"Selected Vosk model: {c(choice['name'], GREEN)}")
    return choice["path"]


# ─── Piper voice catalogue (gender → HuggingFace download info) ───────────────
PIPER_VOICE_CATALOGUE = {
    "male": [
        {
            "name": "en_US-ryan-high",
            "desc": "Ryan (high quality, ~120 MB)",
            "hf_path": "en/en_US/ryan/high",
            "files": ["en_US-ryan-high.onnx", "en_US-ryan-high.onnx.json"],
        },
    ],
    "female": [
        {
            "name": "en_US-amy-high",
            "desc": "Amy (high quality, ~120 MB)",
            "hf_path": "en/en_US/amy/high",
            "files": ["en_US-amy-high.onnx", "en_US-amy-high.onnx.json"],
        },
        {
            "name": "en_US-kathleen-low",
            "desc": "Kathleen (low quality, ~5 MB)",
            "hf_path": "en/en_US/kathleen/low",
            "files": ["en_US-kathleen-low.onnx", "en_US-kathleen-low.onnx.json"],
        },
    ],
}

# Keywords used to guess gender from pyttsx3 voice names
_MALE_KEYWORDS   = {"david", "mark", "daniel", "james", "michael", "george",
                    "thomas", "richard", "male", "man", "guy", "ryan"}
_FEMALE_KEYWORDS = {"zira", "hazel", "victoria", "samantha", "lisa", "karen",
                    "susan", "amy", "alice", "emma", "female", "woman", "girl",
                    "kathleen", "jenny"}


def _download_piper_voice(voice_info: dict) -> str:
    """Download a Piper voice from HuggingFace into piper-voices/. Returns .onnx path."""
    import urllib.request as _ur
    base = ("https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/"
            + voice_info["hf_path"])
    os.makedirs(PIPER_VOICE_DIR, exist_ok=True)
    for fname in voice_info["files"]:
        dest = os.path.join(PIPER_VOICE_DIR, fname)
        if not os.path.isfile(dest):
            info(f"Downloading {fname} …")
            try:
                def _hook(count, block, total):
                    if total > 0:
                        print(f"\r  {int(count*block*100/total)}% ", end="", flush=True)
                _ur.urlretrieve(f"{base}/{fname}", dest, _hook)
                print()
            except Exception as e:
                error(f"Download failed: {e}")
                if os.path.isfile(dest):
                    os.remove(dest)
                sys.exit(1)
    onnx = os.path.join(PIPER_VOICE_DIR, voice_info["files"][0])
    success(f"Piper voice ready: {voice_info['name']}")
    return onnx


def select_tts_voice(args) -> tuple:
    """
    Show a Male / Female picker and return (piper_voice_path, voice_id, tts_mode).
    Handles Piper, pyttsx3, edge-tts, and system TTS.
    Skips the prompt if --gender or --piper-voice / --voice is already supplied.
    """
    tts_mode       = args.tts
    piper_voice_path = getattr(args, "piper_voice", None)
    voice_id       = getattr(args, "voice", None)
    gender_flag    = getattr(args, "gender", None)  # "male" | "female" | None

    # ── If everything was specified on CLI, return immediately ────────────────
    if piper_voice_path and tts_mode == "piper":
        return piper_voice_path, voice_id, tts_mode
    if voice_id and tts_mode != "piper":
        return piper_voice_path, voice_id, tts_mode

    # ── Gender prompt (skip if --gender already given) ────────────────────────
    if gender_flag is None:
        print(f"\n{BOLD}{CYAN}Select output voice gender:{RESET}")
        print(c("─" * 40, DIM))
        print(f"  {c('[1]', CYAN)} {c('♂', CYAN)}  Male")
        print(f"  {c('[2]', CYAN)} {c('♀', MAGENTA)}  Female")
        print(c("─" * 40, DIM))
        print(f"  {c('[Enter]', DIM)} default (Male)")
        print()
        try:
            raw = input(f"{BOLD}Your choice (1/2): {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            raw = ""
        gender = "female" if raw == "2" else "male"
    else:
        gender = gender_flag.lower()

    print()
    success(f"Voice gender: {gender.capitalize()}")

    # ── Piper ─────────────────────────────────────────────────────────────────
    if tts_mode == "piper":
        catalogue = PIPER_VOICE_CATALOGUE[gender]

        # Find already-downloaded voices for this gender
        available = []
        for v in catalogue:
            onnx = os.path.join(PIPER_VOICE_DIR, v["files"][0])
            if os.path.isfile(onnx):
                available.append((v, onnx))

        if len(available) == 1:
            info(f"Using Piper voice: {available[0][0]['name']}")
            return available[0][1], None, tts_mode

        if len(available) > 1:
            # Let user pick among downloaded ones
            print(f"\n{BOLD}{CYAN}Available {gender} Piper voices:{RESET}")
            print(c("─" * 50, DIM))
            for i, (v, _) in enumerate(available, 1):
                print(f"  {c(f'[{i}]', CYAN)} {v['desc']}  {c(v['name'], DIM)}")
            print(c("─" * 50, DIM))
            try:
                raw2 = input(f"{BOLD}Your choice (1-{len(available)}) [Enter=1]: {RESET}").strip()
            except (KeyboardInterrupt, EOFError):
                raw2 = ""
            idx = (int(raw2) - 1) if raw2.isdigit() and 1 <= int(raw2) <= len(available) else 0
            chosen_v, onnx_path = available[idx]
            success(f"Selected: {chosen_v['name']}")
            return onnx_path, None, tts_mode

        # Nothing downloaded yet — download the first option for this gender
        v = catalogue[0]
        print(f"{YELLOW}No {gender} Piper voice found locally.{RESET}")
        info(f"Downloading {v['desc']} …")
        onnx_path = _download_piper_voice(v)
        return onnx_path, None, tts_mode

    # ── edge-tts ──────────────────────────────────────────────────────────────
    if tts_mode == "edge":
        edge_voice = "en-US-GuyNeural" if gender == "male" else "en-US-JennyNeural"
        info(f"edge-tts voice: {edge_voice}")
        return None, edge_voice, tts_mode

    # ── pyttsx3 / system ──────────────────────────────────────────────────────
    if tts_mode in ("pyttsx3", "system") and PYTTSX3_AVAILABLE:
        try:
            import pyttsx3 as _px
            _eng = _px.init()
            all_voices = _eng.getProperty("voices")
            _eng.stop()
            del _eng

            def _gender_of(v):
                tokens = (v.name + " " + " ".join(v.languages)).lower()
                if any(k in tokens for k in _FEMALE_KEYWORDS):
                    return "female"
                if any(k in tokens for k in _MALE_KEYWORDS):
                    return "male"
                return "unknown"

            filtered = [v for v in all_voices if _gender_of(v) == gender]
            if not filtered:
                warn(f"No {gender} pyttsx3 voices found — using system default.")
                return None, None, tts_mode

            if len(filtered) == 1:
                info(f"pyttsx3 voice: {filtered[0].name}")
                return None, filtered[0].id, tts_mode

            # Multiple matches — let user pick
            print(f"\n{BOLD}{CYAN}Available {gender} system voices:{RESET}")
            print(c("─" * 50, DIM))
            for i, v in enumerate(filtered, 1):
                print(f"  {c(f'[{i}]', CYAN)} {v.name}  {c(v.id, DIM)}")
            print(c("─" * 50, DIM))
            try:
                raw3 = input(f"{BOLD}Your choice (1-{len(filtered)}) [Enter=1]: {RESET}").strip()
            except (KeyboardInterrupt, EOFError):
                raw3 = ""
            idx3 = (int(raw3) - 1) if raw3.isdigit() and 1 <= int(raw3) <= len(filtered) else 0
            success(f"Selected: {filtered[idx3].name}")
            return None, filtered[idx3].id, tts_mode
        except Exception as e:
            warn(f"Could not enumerate pyttsx3 voices ({e}) — using default.")

    return None, None, tts_mode


# ─── Entry Point ───────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Local AI Voice Assistant powered by Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python assistant.py
  python assistant.py --model mistral
  python assistant.py --model llama3 --tts edge --edge-voice en-US-JennyNeural
  python assistant.py --wake-word jarvis --rate 160
  python assistant.py --list-voices
        """
    )
    parser.add_argument("--model",       default=None,
                        help="Ollama model to use (omit to pick interactively from installed models)")
    parser.add_argument("--wake-word",   default=DEFAULT_WAKE_WORD, help=f"Wake word to listen for (default: {DEFAULT_WAKE_WORD})")
    parser.add_argument("--tts",         choices=["piper", "pyttsx3", "edge", "none"],
                        default=DEFAULT_TTS, help="TTS engine (default: piper → pyttsx3 → system fallback)")
    parser.add_argument("--gender",      choices=["male", "female"], default=None,
                        help="Voice gender to use (male/female). Omit to pick interactively.")
    parser.add_argument("--voice",       default=None, help="Exact voice ID for pyttsx3 or voice name for edge-tts (overrides --gender)")
    parser.add_argument("--edge-voice",  default=None, help="edge-tts voice name (overrides --gender for edge TTS)")
    parser.add_argument("--piper-voice", default=None, metavar="PATH",
                        help="Path to a Piper .onnx voice file (overrides --gender for Piper TTS)")
    parser.add_argument("--rate",        type=int, default=175, help="Speech rate for pyttsx3 (default: 175)")
    parser.add_argument("--list-voices", action="store_true", help="List available TTS voices and exit")
    parser.add_argument("--vosk-model",  default=None, metavar="PATH",
                        help="Path to a Vosk model directory (auto-downloads small English model if omitted)")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Model selection (interactive if not given) ────────────────────────────
    chosen_llm_model = select_ollama_model(args.model)
    chosen_vosk_path = select_vosk_model(args.vosk_model)

    # ── Voice gender selection ────────────────────────────────────────────────
    piper_voice_path, voice_id, args.tts = select_tts_voice(args)

    # Fallback: if Piper mode but no voice resolved, drop to pyttsx3
    if args.tts == "piper" and not piper_voice_path:
        warn("No Piper voice resolved — falling back to pyttsx3.")
        args.tts = "pyttsx3"

    assistant = VoiceAssistant(
        model            = chosen_llm_model,
        tts_mode         = args.tts,
        voice_id         = voice_id,
        tts_rate         = args.rate,
        wake_word        = args.wake_word,
        vosk_model_path  = chosen_vosk_path,
        piper_voice_path = piper_voice_path,
    )

    if args.list_voices:
        assistant.tts.list_voices()
        sys.exit(0)

    assistant.run()


if __name__ == "__main__":
    main()
