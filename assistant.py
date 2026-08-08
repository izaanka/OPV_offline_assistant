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
    import pygame.mixer
    pygame.mixer.init()
    pygame.mixer.quit()
    PYGAME_AVAILABLE = True
except Exception:
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
except Exception:
    PYAUDIO_AVAILABLE = False

try:
    import sounddevice as _sd_check
    import numpy as _np_check
    SOUNDDEVICE_AVAILABLE = True
except Exception:
    SOUNDDEVICE_AVAILABLE = False

if not PYAUDIO_AVAILABLE and not SOUNDDEVICE_AVAILABLE:
    error("No working audio input library found.")
    error("Please install system package: sudo apt install portaudio19-dev espeak")
    error("Or run launcher: ./run_assistant.sh")
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

SYSTEM_PROMPT = """You are a warm, friendly, and empathetic companion and voice assistant.
Speak casually and naturally like a real human friend — relaxed, conversational, and concise (aim for 1-2 short natural sentences).
NEVER use robotic AI clichés like "functioning properly", "I don't have feelings like humans do", "as an AI", or formal corporate scripts.
If you know the user's name or facts from long-term memory, address them by their name naturally in conversation.
Do not use markdown formatting, bullet points, or special symbols."""

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
            try:
                pygame.mixer.init()
                pygame.mixer.music.load(path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    if self._stop_event.is_set():
                        pygame.mixer.music.stop()
                        return False
                    time.sleep(0.05)
                return True
            except Exception as e:
                warn(f"PyGame playback error ({e}) — falling back to system audio player.")

        # Fallback: system player
        _os = _platform.system()
        if _os == "Darwin":
            cmd = f"afplay '{path}'"
        elif _os == "Windows":
            cmd = f'powershell -NoProfile -Command "(New-Object System.Media.SoundPlayer \'{path}\').PlaySync()"'
        else:
            if path.endswith(".wav"):
                cmd = f"aplay -q '{path}' 2>/dev/null || paplay '{path}' 2>/dev/null || ffplay -nodisp -autoexit '{path}' 2>/dev/null"
            else:
                cmd = f"mpg123 -q '{path}' 2>/dev/null || ffplay -nodisp -autoexit '{path}' 2>/dev/null"
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

    def listen_once_timeout(self, timeout: float = 5.0) -> Optional[str]:
        """Listen for one utterance with custom timeout."""
        if self._use_pyaudio:
            return self._listen_pyaudio(timeout, self.phrase_limit)
        else:
            return self._listen_sounddevice(timeout, self.phrase_limit)

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

# ─── Phonetic Aliases & City/Name Gazetteer Normalization ───────────────────────
ALIASES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".aliases.json")

DEFAULT_ALIASES = {
    "is on": "Izaan",
    "is an": "Izaan",
    "eyes on": "Izaan",
    "ezan": "Izaan",
    "izan": "Izaan",
    "daily": "Delhi",
    "dilli": "Delhi",
    "del hi": "Delhi",
    "bomb bay": "Mumbai",
    "mombay": "Mumbai",
    "bang a door": "Bengaluru",
    "bangalore": "Bengaluru",
    "bengaluru": "Bengaluru",
    "call cut a": "Kolkata",
    "kolkata": "Kolkata"
}

GLOBAL_CITIES = [
    "Delhi", "Mumbai", "Bengaluru", "Kolkata", "Chennai", "Hyderabad", "Ahmedabad", "Pune",
    "Jaipur", "Surat", "Lucknow", "Kanpur", "Nagpur", "Indore", "Bhopal", "Patna", "Vadodara",
    "Ludhiana", "Agra", "Nashik", "Faridabad", "Meerut", "Rajkot", "Kalyan", "Varanasi",
    "Srinagar", "Aurangabad", "Dhanbad", "Amritsar", "Navi Mumbai", "Allahabad", "Ranchi",
    "Howrah", "Coimbatore", "Jabalpur", "Gwalior", "Vijayawada", "Jodhpur", "Madurai",
    "Raipur", "Kota", "Guwahati", "Chandigarh", "Solapur", "Hubli", "Bareilly", "Moradabad",
    "Mysore", "Gurgaon", "Aligarh", "Jalandhar", "Tiruchirappalli", "Bhubaneswar", "Salem",
    "Warangal", "Mira-Bhayandar", "Thiruvananthapuram", "Bhiwandi", "Saharanpur", "Guntur",
    "Amravati", "Bikaner", "Noida", "Jamshedpur", "Bhilai", "Cuttack", "Firozabad", "Kochi",
    "Bhavnagar", "Dehradun", "Durgapur", "Asansol", "Nanded", "Kolhapur", "Ajmer", "Gulbarga",
    "Jamnagar", "Ujjain", "Loni", "Siliguri", "Jhansi", "Ulhasnagar", "Jammu", "Sangli",
    "Mangalore", "Erode", "Belgaum", "Ambattur", "Tirunelveli", "Malegaon", "Gaya", "Jalgaon",
    "Udaipur", "London", "Paris", "New York", "Tokyo", "Berlin", "Beijing", "Shanghai",
    "Sydney", "Toronto", "Dubai", "Singapore", "Rome", "Madrid", "Chicago", "Los Angeles",
    "San Francisco", "Seattle", "Washington", "Boston", "Moscow", "Seoul", "Bangkok"
]
CITY_LOWER_MAP = {c.lower(): c for c in GLOBAL_CITIES}

def load_aliases() -> dict:
    """Load phonetic sound-alike mappings from .aliases.json."""
    aliases = dict(DEFAULT_ALIASES)
    if os.path.isfile(ALIASES_FILE):
        try:
            with open(ALIASES_FILE, "r", encoding="utf-8") as f:
                custom = json.load(f)
                if isinstance(custom, dict):
                    aliases.update(custom)
        except Exception as e:
            warn(f"Could not load .aliases.json ({e})")
    return aliases

def save_alias(phrase: str, correction: str):
    """Save a custom sound-alike phrase to .aliases.json."""
    aliases = load_aliases()
    aliases[phrase.lower().strip()] = correction.strip()
    try:
        with open(ALIASES_FILE, "w", encoding="utf-8") as f:
            json.dump(aliases, f, indent=2)
    except Exception as e:
        warn(f"Could not save .aliases.json ({e})")

def normalize_utterance(text: str) -> str:
    """Post-process raw speech-to-text output using phonetic aliases & city gazetteer fuzzy matching."""
    if not text:
        return text

    import re
    import difflib

    result = text

    # 1. Apply phonetic aliases
    aliases = load_aliases()
    for phrase, correction in aliases.items():
        pattern = re.compile(r'\b' + re.escape(phrase) + r'\b', re.IGNORECASE)
        result = pattern.sub(correction, result)

    # 2. Fuzzy city correction after prepositions (in, at, near, for, weather, climate, from)
    words = result.split()
    corrected_words = []
    i = 0
    while i < len(words):
        word = words[i]
        clean_w = re.sub(r'[^a-zA-Z]', '', word).lower()
        prev = words[i-1].lower() if i > 0 else ''
        if prev in ('in', 'at', 'near', 'for', 'weather', 'climate', 'from') and len(clean_w) >= 3:
            if clean_w in CITY_LOWER_MAP:
                corrected_words.append(CITY_LOWER_MAP[clean_w])
                i += 1
                continue
            matches = difflib.get_close_matches(clean_w, CITY_LOWER_MAP.keys(), n=1, cutoff=0.7)
            if matches:
                corrected_words.append(CITY_LOWER_MAP[matches[0]])
                i += 1
                continue
        corrected_words.append(word)
        i += 1

    return ' '.join(corrected_words)


    def _transcribe(self, audio, recognizer=None) -> Optional[str]:
        """Transcribe audio to text using Vosk and normalize phonetics/cities."""
        try:
            raw = audio.get_raw_data(convert_rate=16000, convert_width=2)
            rec = vosk.KaldiRecognizer(self._vosk_model, 16000)
            rec.AcceptWaveform(raw)
            result = json.loads(rec.FinalResult())
            text = result.get("text", "").strip()
            if text:
                normalized = normalize_utterance(text)
                if normalized != text:
                    info(f"Phonetic normalized: '{text}' → '{normalized}'")
                return normalized
            return None
        except Exception as e:
            warn(f"Vosk transcription error: {e}")
            return None

    def contains_wake_word(self, text: str, wake_word: str) -> bool:
        return wake_word.lower() in text.lower().split()


# ─── Long-term Memory (.memory) ───────────────────────────────────────────────
MEMORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".memory")

def load_memory() -> list:
    """Load list of remembered facts from .memory."""
    if os.path.isfile(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception as e:
            warn(f"Could not load .memory ({e})")
    return []

def save_memory(facts: list):
    """Save list of remembered facts to .memory."""
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(facts, f, indent=2)
    except Exception as e:
        warn(f"Could not save .memory ({e})")

def add_memory_fact(fact: str) -> bool:
    """Add a new fact to .memory file."""
    facts = load_memory()
    clean = fact.strip()
    if clean and clean not in facts:
        facts.append(clean)
        save_memory(facts)
        return True
    return False

def auto_extract_facts(text: str):
    """Automatically detect personal facts (name, location, preferences) and save to .memory."""
    import re
    text_clean = text.strip()

    # Name detection (e.g. "my name is Izaan", "i'm Izaan", "call me Izaan")
    m_name = re.search(r"\b(?:my name is|call me|i'm|i am)\s+([A-Z][a-z]+|[a-z]+)\b", text_clean, re.IGNORECASE)
    if m_name:
        name = m_name.group(1).capitalize()
        ignore = {"good", "fine", "ok", "okay", "happy", "sad", "tired", "busy", "relaxing", "just", "doing", "here", "ready", "trying", "sure", "not", "great", "awesome", "cool", "back", "done", "alone", "bored", "late", "well"}
        if name.lower() not in ignore and len(name) > 1:
            if add_memory_fact(f"User's name is {name}"):
                info(f"Auto-remembered: User's name is {name}")

    # Location detection (e.g. "i live in Delhi", "i'm from London")
    m_loc = re.search(r"\b(?:i live in|i'm from|i am from)\s+([a-zA-Z\s]+)\b", text_clean, re.IGNORECASE)
    if m_loc:
        loc = m_loc.group(1).strip().title()
        if len(loc) > 2 and add_memory_fact(f"User lives in {loc}"):
            info(f"Auto-remembered: User lives in {loc}")

    # Favorites / Likes detection (e.g. "my favorite language is Python", "i love pizza")
    m_like = re.search(r"\b(?:my favorite|i love|i really like)\s+([a-zA-Z0-9\s]{3,30})\b", text_clean, re.IGNORECASE)
    if m_like:
        fav = m_like.group(1).strip()
        if add_memory_fact(f"User likes {fav}"):
            info(f"Auto-remembered: User likes {fav}")


# ─── Online Knowledge & Web Tools ──────────────────────────────────────────────

def fetch_online_weather(location: str = "auto") -> Optional[str]:
    """Fetch live weather data from wttr.in for a given location or auto IP location."""
    import urllib.parse
    import urllib.request
    try:
        loc_clean = location.strip() if location and location.strip().lower() != "auto" else ""
        loc_encoded = urllib.parse.quote(loc_clean)
        url = f"https://wttr.in/{loc_encoded}?format=j1"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="ignore"))
            curr = data["current_condition"][0]
            area = data["nearest_area"][0]["areaName"][0]["value"]
            country = data["nearest_area"][0]["country"][0]["value"]
            temp_c = curr["temp_C"]
            temp_f = curr["temp_F"]
            desc = curr["weatherDesc"][0]["value"]
            humidity = curr["humidity"]
            wind = curr["windspeedKmph"]
            return f"Current weather in {area}, {country}: {desc}, {temp_c}°C ({temp_f}°F), Humidity: {humidity}%, Wind: {wind} km/h."
    except Exception as e:
        warn(f"Weather lookup note: {e}")
        return None


def fetch_universal_web_search(query: str) -> Optional[str]:
    """Search the web as a whole for any topic using DuckDuckGo + Wikipedia APIs."""
    import urllib.parse
    import urllib.request
    import html
    import re

    results = []
    clean_q = query.strip()

    # 1. DuckDuckGo Web Search (POST method for reliable HTML search results)
    try:
        url = "https://html.duckduckgo.com/html/"
        data = urllib.parse.urlencode({"q": clean_q}).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://html.duckduckgo.com/"
        })
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
            titles = re.findall(r'<a class="result__a"[^>]*>(.*?)</a>', body, re.DOTALL)
            snippets = re.findall(r'<a class="result__snippet[^"]*"[^>]*>(.*?)</a>', body, re.DOTALL)
            for t, s in zip(titles[:3], snippets[:3]):
                t_clean = html.unescape(re.sub(r'<[^>]+>', '', t)).strip()
                s_clean = html.unescape(re.sub(r'<[^>]+>', '', s)).strip()
                if t_clean and s_clean:
                    results.append(f"• {t_clean}: {s_clean}")
    except Exception as e:
        warn(f"Web search note ({e})")

    # 2. Wikipedia Search API fallback / supplement for knowledge queries
    if len(results) < 2:
        try:
            url_wiki = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(clean_q)}&format=json"
            req_wiki = urllib.request.Request(url_wiki, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req_wiki, timeout=4) as resp:
                data_wiki = json.loads(resp.read().decode("utf-8", errors="ignore"))
                items = data_wiki.get("query", {}).get("search", [])
                for item in items[:2]:
                    title = item.get("title", "")
                    snippet = html.unescape(re.sub(r'<[^>]+>', '', item.get("snippet", "")))
                    if title and snippet:
                        results.append(f"• Wikipedia ({title}): {snippet}")
        except Exception:
            pass

    if results:
        return "\n".join(results)
    return None


def get_online_context(user_input: str) -> Optional[str]:
    """Detect if user query requires live weather or general web search as a whole."""
    text_lower = user_input.lower().strip()
    import re

    # 1. Weather & Climate detection
    weather_keywords = ["weather", "climate", "temperature", "forecast", "how hot", "how cold", "rain", "rainy", "snow", "sunny", "humidity"]
    if any(k in text_lower for k in weather_keywords):
        m = re.search(r"\b(?:in|at|for|near)\s+([a-zA-Z\s]+)\b", user_input, re.IGNORECASE)
        location = m.group(1).strip() if m else "auto"
        for kw in ["today", "right now", "tomorrow", "this week", "currently"]:
            location = location.split(kw)[0].strip()
        info(f"🌐 Fetching online weather for '{location}'...")
        weather_info = fetch_online_weather(location)
        if weather_info:
            success("Retrieved live weather data!")
            return f"Live Real-Time Weather Data:\n{weather_info}"

    # 2. Universal Web Search as a whole for knowledge, news, facts, or questions
    skip_phrases = {
        "hi", "hello", "hey", "how are you", "thanks", "thank you", "bye", "goodbye",
        "quit", "exit", "stop", "reset", "clear memory", "show memory", "what do you remember",
        "i'm good", "i'm fine", "doing well", "nothing much", "just relaxing", "no thanks", "yes", "no", "okay", "ok"
    }
    if text_lower in skip_phrases:
        return None

    web_triggers = [
        "what", "who", "where", "when", "why", "how", "is ", "are ", "can ", "could ",
        "tell me", "search", "lookup", "find", "news", "price", "score", "latest",
        "recent", "today", "history", "definition", "explain", "meaning"
    ]
    should_search = any(text_lower.startswith(t) or f" {t}" in text_lower for t in web_triggers) or ("?" in text_lower) or (len(text_lower.split()) >= 3)

    if should_search:
        info(f"🌐 Searching the web for '{user_input}'...")
        search_info = fetch_universal_web_search(user_input)
        if search_info:
            success("Retrieved live web search results!")
            return f"Live Real-Time Web Search Results:\n{search_info}"

    return None


# ─── Ollama Client ─────────────────────────────────────────────────────────────
class OllamaLLM:
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model   = model
        self.history = []  # list of {"role": ..., "content": ...}

    def reset(self):
        self.history.clear()

    def _get_system_prompt(self) -> str:
        prompt = SYSTEM_PROMPT
        facts = load_memory()
        if facts:
            facts_str = "\n".join(f"- {f}" for f in facts)
            prompt += f"\n\nLong-term Memory (Stored User Facts):\n{facts_str}"
        return prompt

    def warmup(self):
        """Warm up model in VRAM/RAM with a small 1-token prompt for instant TTFT."""
        info("Pre-warming Ollama LLM into memory for ultra-fast response...")
        try:
            if OLLAMA_LIB_AVAILABLE:
                ollama_client.generate(model=self.model, prompt="hi", options={"num_predict": 1})
            else:
                payload = _json.dumps({
                    "model": self.model,
                    "prompt": "hi",
                    "options": {"num_predict": 1},
                    "stream": False
                }).encode()
                req = urllib.request.Request(
                    "http://localhost:11434/api/generate",
                    data=payload,
                    headers={"Content-Type": "application/json"}
                )
                urllib.request.urlopen(req, timeout=10)
            success("LLM pre-warmed & ready in memory!")
        except Exception as e:
            warn(f"LLM pre-warm note ({e})")

    def _trim_history(self):
        if len(self.history) > CONVERSATION_HIST * 2:
            self.history = self.history[-(CONVERSATION_HIST * 2):]

    def chat(self, user_input: str) -> str:
        # Check if online context (weather, news, search) is needed
        online_ctx = get_online_context(user_input)

        effective_input = user_input
        if online_ctx:
            effective_input = f"{user_input}\n\n[{online_ctx}]"

        self.history.append({"role": "user", "content": effective_input})
        self._trim_history()

        messages = [{"role": "system", "content": self._get_system_prompt()}] + self.history

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
             {c('say "remember [fact]" to store long-term memory', DIM)}
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

        # Warm up Ollama LLM into GPU/RAM for instant TTFT
        self.llm.warmup()

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
        """Capture command and enter 30-second conversational follow-up window."""
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

        # Process initial query
        if not self._process_and_respond(user_input):
            return

        # ── 30-Second Conversational Follow-Up Window ───────────────────────────
        # Listens directly for questions without needing to say "hey" after every answer!
        FOLLOWUP_WINDOW = 30.0
        followup_start  = time.time()

        while (time.time() - followup_start < FOLLOWUP_WINDOW) and self._running:
            remaining = int(FOLLOWUP_WINDOW - (time.time() - followup_start))
            print(f"\r\033[K{c(f'💬 Conversational follow-up active ({remaining}s remaining — speak directly)...', CYAN)}", end="", flush=True)

            text = self.listener.listen_once_timeout(timeout=4.0)
            if not text or not text.strip():
                continue

            text = text.strip()
            print("\r\033[K", end="", flush=True)  # Clear status line before printing user label

            # Strip wake word if repeated in follow-up mode
            words = text.lower().split()
            if self.wake_word in words:
                idx  = words.index(self.wake_word)
                text = " ".join(words[idx + 1:]).strip()
                if not text:
                    self.tts.speak("Yes?")
                    followup_start = time.time()
                    continue

            user_label(text)
            should_continue = self._process_and_respond(text)
            if should_continue:
                # Reset 30s timer after replying so user can continue asking follow-ups!
                followup_start = time.time()
            else:
                break

        print(f"\r\033[K\n{BOLD}Listening for wake word: \"{self.wake_word}\" ...{RESET}\n")

    def _process_and_respond(self, user_input: str) -> bool:
        """Process user input, handle special/memory commands, auto-extract facts, and chat. Returns False if exiting."""
        cmd = user_input.lower().strip()

        # Auto-extract any facts (like name, location, likes) automatically from user speech
        auto_extract_facts(user_input)

        # Special commands
        if cmd in ("quit", "exit", "bye", "goodbye"):
            self._running = False
            return False
        if cmd in ("reset", "clear history", "start over"):
            self.llm.reset()
            success("Conversation history cleared.")
            self.tts.speak("Okay, starting fresh!")
            return True

        # Memory commands
        if cmd.startswith("remember "):
            fact = user_input[9:].strip()
            if fact:
                add_memory_fact(fact)
                success(f"Saved to .memory: {fact}")
                self.tts.speak("I have saved that to my long term memory.")
            return True

        # Alias command (e.g. "alias daily as Delhi", "alias is on as Izaan")
        if cmd.startswith("alias ") and " as " in cmd:
            parts = user_input[6:].split(" as ", 1)
            if len(parts) == 2:
                phrase, correction = parts[0].strip(), parts[1].strip()
                if phrase and correction:
                    save_alias(phrase, correction)
                    success(f"Saved sound-alike alias: '{phrase}' → '{correction}'")
                    self.tts.speak(f"Got it! I will now recognize {phrase} as {correction}.")
                    return True
        if cmd in ("clear memory", "forget memory", "reset memory"):
            save_memory([])
            success("Long-term memory cleared.")
            self.tts.speak("Long term memory cleared.")
            return True
        if cmd in ("what do you remember", "show memory", "list memory"):
            facts = load_memory()
            if facts:
                facts_text = ". ".join(facts)
                info(f"Memory: {facts_text}")
                self.tts.speak(f"Here is what I remember: {facts_text}")
            else:
                info("Memory is empty.")
                self.tts.speak("I don't have any saved memories yet.")
            return True

        # Query LLM
        info("Thinking...")
        response = self.llm.chat(user_input)
        ai_label(response)

        # Speak response (interruptible with "stop")
        self._speak_with_stop_listener(response)
        return True

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
        "name": "vosk-model-en-us-0.22-lgraph",
        "desc": "Medium English (128 MB) — Balanced accuracy & speed [Default]",
        "localZip": _LGRAPH_LOCAL_ZIP,
        "url": "https://alphacephei.com/vosk/models/vosk-model-en-us-0.22-lgraph.zip"
    },
    {
        "name": "vosk-model-en-us-daanzu-20200905",
        "desc": "Daanzu Dictation (920 MB / ~500 MB graph) — ★ High Accuracy for Dictation & Speech",
        "url": "https://alphacephei.com/vosk/models/vosk-model-en-us-daanzu-20200905.zip"
    },
    {
        "name": "vosk-model-en-us-librispeech-0.2",
        "desc": "LibriSpeech English (845 MB) — ★ High Accuracy (845MB LibriSpeech dataset)",
        "url": "https://alphacephei.com/vosk/models/vosk-model-en-us-librispeech-0.2.zip"
    },
    {
        "name": "vosk-model-en-us-0.22",
        "desc": "Large English (1.8 GB) — ★ Highest Accuracy (Full 128k vocabulary)",
        "url": "https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip"
    },
    {
        "name": "vosk-model-en-us-0.42-gigaspeech",
        "desc": "GigaSpeech English (2.3 GB) — ★ High Accuracy (Trained on 14,000h Gigaspeech)",
        "url": "https://alphacephei.com/vosk/models/vosk-model-en-us-0.42-gigaspeech.zip"
    },
    {
        "name": "vosk-model-en-in-0.5",
        "desc": "Large Indian English (1.0 GB) — ★ High Accuracy for Indian Accents",
        "url": "https://alphacephei.com/vosk/models/vosk-model-en-in-0.5.zip"
    },
    {
        "name": "vosk-model-small-en-in-0.4",
        "desc": "Small Indian English (36 MB) — Fast, tuned for Indian Accents",
        "url": "https://alphacephei.com/vosk/models/vosk-model-small-en-in-0.4.zip"
    },
    {
        "name": "vosk-model-small-en-us-0.15",
        "desc": "Small English (40 MB) — Lightweight for low-power CPUs",
        "url": "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
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
            "name": "en_US-ryan-medium",
            "desc": "Ryan (Medium, ~60 MB) — ★ Balanced speed & naturalness [Default]",
            "hf_path": "en/en_US/ryan/medium",
            "files": ["en_US-ryan-medium.onnx", "en_US-ryan-medium.onnx.json"],
        },
        {
            "name": "en_US-ryan-high",
            "desc": "Ryan (Large & High Quality, ~120 MB) — Deep, studio quality male voice",
            "hf_path": "en/en_US/ryan/high",
            "files": ["en_US-ryan-high.onnx", "en_US-ryan-high.onnx.json"],
        },
        {
            "name": "en_US-danny-low",
            "desc": "Danny (Small & Fast, ~15 MB) — Ultra-low latency voice",
            "hf_path": "en/en_US/danny/low",
            "files": ["en_US-danny-low.onnx", "en_US-danny-low.onnx.json"],
        },
    ],
    "female": [
        {
            "name": "en_US-libritts_r-medium",
            "desc": "LibriTTS (Medium, ~60 MB) — ★ Natural expressive female voice [Default]",
            "hf_path": "en/en_US/libritts_r/medium",
            "files": ["en_US-libritts_r-medium.onnx", "en_US-libritts_r-medium.onnx.json"],
        },
        {
            "name": "en_US-amy-medium",
            "desc": "Amy (Medium, ~60 MB) — Balanced speed & high naturalness",
            "hf_path": "en/en_US/amy/medium",
            "files": ["en_US-amy-medium.onnx", "en_US-amy-medium.onnx.json"],
        },
        {
            "name": "en_US-lessac-high",
            "desc": "Lessac (Large & High Quality, ~120 MB) — Clear, studio quality female voice",
            "hf_path": "en/en_US/lessac/high",
            "files": ["en_US-lessac-high.onnx", "en_US-lessac-high.onnx.json"],
        },
        {
            "name": "en_US-kathleen-low",
            "desc": "Kathleen (Small & Fast, ~15 MB) — Ultra-low latency voice",
            "hf_path": "en/en_US/kathleen/low",
            "files": ["en_US-kathleen-low.onnx", "en_US-kathleen-low.onnx.json"],
        },
    ],
}

# Keywords used to guess gender from pyttsx3 voice names
_MALE_KEYWORDS   = {"david", "mark", "daniel", "james", "michael", "george",
                    "thomas", "richard", "male", "man", "guy", "ryan", "danny"}
_FEMALE_KEYWORDS = {"zira", "hazel", "victoria", "samantha", "lisa", "karen",
                    "susan", "amy", "alice", "emma", "female", "woman", "girl",
                    "kathleen", "jenny", "lessac", "libritts"}


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
                        print(f"\r  Progress: {int(count*block*100/total)}% ", end="", flush=True)
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
    Show a Male / Female picker and a Voice Model quality picker, returning (piper_voice_path, voice_id, tts_mode).
    Handles Piper, pyttsx3, edge-tts, and system TTS.
    Skips the prompt if --piper-voice or --voice is already supplied on CLI.
    """
    tts_mode       = args.tts
    piper_voice_path = getattr(args, "piper_voice", None)
    voice_id       = getattr(args, "voice", None)
    gender_flag    = getattr(args, "gender", None)  # "male" | "female" | None

    # ── If everything was specified on CLI or loaded from .config, return immediately ────────────────
    if piper_voice_path and tts_mode == "piper" and os.path.isfile(piper_voice_path):
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
        print(f"  {c('[Enter]', DIM)} default (Female — LibriTTS)")
        print()
        try:
            raw = input(f"{BOLD}Your choice (1/2): {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            raw = ""
        gender = "male" if raw == "1" else "female"
    else:
        gender = gender_flag.lower()

    print()
    success(f"Voice gender: {gender.capitalize()}")

    # ── Piper Voice Model Quality Picker ──────────────────────────────────────
    if tts_mode == "piper":
        catalogue = PIPER_VOICE_CATALOGUE[gender]

        print(f"\n{BOLD}{CYAN}Select a {gender.capitalize()} Piper voice model (Speed / Quality):{RESET}")
        print(c("─" * 65, DIM))

        choices = []
        default_idx = 1
        for i, v in enumerate(catalogue, 1):
            onnx_path = os.path.join(PIPER_VOICE_DIR, v["files"][0])
            downloaded = os.path.isfile(onnx_path)
            status = c("[Ready]", GREEN) if downloaded else c("[Download required]", YELLOW)
            if "libritts" in v["name"].lower():
                default_idx = i
            choices.append((v, onnx_path, downloaded))
            print(f"  {c(f'[{i}]', CYAN)} {v['desc']}  {status}")
            print(f"      {c(v['name'], DIM)}")
        print(c("─" * 65, DIM))
        print(f"  {c('[Enter]', DIM)} use default ({default_idx})")
        print()

        try:
            raw2 = input(f"{BOLD}Your choice (1-{len(choices)}): {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            raw2 = ""

        idx = (int(raw2) - 1) if raw2.isdigit() and 1 <= int(raw2) <= len(choices) else (default_idx - 1)
        chosen_v, onnx_path, is_ready = choices[idx]

        if not is_ready:
            info(f"Downloading {chosen_v['desc']} …")
            onnx_path = _download_piper_voice(chosen_v)

        success(f"Selected Piper voice model: {chosen_v['name']}")
        args.gender = gender
        args.piper_voice = onnx_path
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
            args.gender = gender
            return None, filtered[idx3].id, tts_mode
        except Exception as e:
            warn(f"Could not enumerate pyttsx3 voices ({e}) — using default.")

    args.gender = gender
    return None, None, tts_mode


# ─── Configuration Persistence (.config) ───────────────────────────────────────
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".config")

def load_config() -> dict:
    """Load configuration from .config file if present."""
    if os.path.isfile(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception as e:
            warn(f"Could not load configuration from .config ({e})")
    return {}

def save_config(cfg: dict):
    """Save configuration to .config file."""
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        success(f"Saved active configuration to .config")
    except Exception as e:
        warn(f"Could not save configuration to .config ({e})")


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
  python assistant.py --reconfigure
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
    parser.add_argument("--reconfigure", action="store_true", help="Reset saved options in .config and re-select interactively")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config() if not args.reconfigure else {}

    # Restore saved options if CLI flags were omitted
    if not args.model and config.get("model"):
        args.model = config["model"]
        info(f"Loaded saved LLM model from .config: {args.model}")

    if not args.vosk_model and config.get("vosk_model"):
        if os.path.exists(config["vosk_model"]):
            args.vosk_model = config["vosk_model"]
            info(f"Loaded saved Vosk model from .config: {args.vosk_model}")

    if not args.piper_voice and config.get("piper_voice"):
        if os.path.exists(config["piper_voice"]):
            args.piper_voice = config["piper_voice"]
            info(f"Loaded saved Piper voice model from .config: {os.path.basename(args.piper_voice)}")

    if not args.gender and config.get("gender"):
        args.gender = config["gender"]
        info(f"Loaded saved voice gender from .config: {args.gender}")

    if not args.wake_word or args.wake_word == DEFAULT_WAKE_WORD:
        if config.get("wake_word"):
            args.wake_word = config["wake_word"]

    # ── Model selection (interactive if not given/saved) ───────────────────────
    chosen_llm_model = select_ollama_model(args.model)
    chosen_vosk_path = select_vosk_model(args.vosk_model)

    # ── Voice gender selection ────────────────────────────────────────────────
    piper_voice_path, voice_id, args.tts = select_tts_voice(args)

    # Fallback: if Piper mode but no voice resolved, drop to pyttsx3
    if args.tts == "piper" and not piper_voice_path:
        warn("No Piper voice resolved — falling back to pyttsx3.")
        args.tts = "pyttsx3"

    # Save active configuration to .config for future bootups
    new_config = {
        "model": chosen_llm_model,
        "vosk_model": chosen_vosk_path,
        "tts": args.tts,
        "gender": getattr(args, "gender", None),
        "voice": voice_id,
        "piper_voice": piper_voice_path,
        "wake_word": args.wake_word,
        "rate": args.rate
    }
    save_config(new_config)

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

