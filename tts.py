"""TTS module for OPV voice assistant.

Supports Piper (neural, offline), pyttsx3, Edge-TTS, and System Native TTS.
Audio playback is interruptible and falls back gracefully across platforms.
"""

import io
import os
import time
import wave
import tempfile
import subprocess
import threading
import platform as _platform
from typing import Optional, List, Any

from utils import c, info, success, warn, error, speak_label, BOLD, CYAN, GREEN, YELLOW, RED, MAGENTA, DIM, RESET
from config import PIPER_VOICE_DIR

# ─── Optional Dependencies ─────────────────────────────────────────────────────

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

try:
    import edge_tts
    import asyncio
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False

# Safely check pygame.mixer (some Python builds lack SDL_mixer)
PYGAME_AVAILABLE = False
try:
    import pygame
    import pygame.mixer
    pygame.mixer.init()
    pygame.mixer.quit()
    PYGAME_AVAILABLE = True
except Exception:
    PYGAME_AVAILABLE = False

try:
    from piper.voice import PiperVoice
    PIPER_AVAILABLE = True
except ImportError:
    PIPER_AVAILABLE = False


# ─── TTSEngine Class ───────────────────────────────────────────────────────────

class TTSEngine:
    """Text-to-speech engine supporting Piper, pyttsx3, Edge-TTS, and system native backends."""

    def __init__(self, mode: str = "piper", voice_id: Optional[str] = None, rate: int = 175,
                 piper_voice_path: Optional[str] = None):
        self.mode = mode.lower()
        self.rate = rate
        self.voice_id = voice_id
        self._stop_event = threading.Event()
        self._is_speaking = False
        self._lock = threading.Lock()

        self.engine = None
        self.piper_voice = None
        self._sys_proc: Optional[subprocess.Popen] = None

        self.piper_voice_path = piper_voice_path
        self._init_engine()

    def _init_engine(self):
        """Initialize TTS engine model resources."""
        # ── Piper Neural TTS (Primary) ───────────────────────────────────────
        if self.mode == "piper":
            if PIPER_AVAILABLE and self.piper_voice_path and os.path.isfile(self.piper_voice_path):
                try:
                    info("Loading Piper neural TTS voice model...")
                    config_path = self.piper_voice_path + ".json"
                    self.piper_voice = PiperVoice.load(
                        self.piper_voice_path,
                        config_path=config_path if os.path.isfile(config_path) else None,
                        use_cuda=False
                    )
                    success("Piper TTS ready (neural, 100% offline).")
                except Exception as e:
                    warn(f"Failed to load Piper voice ({e}) — falling back to pyttsx3")
                    self.mode = "pyttsx3"
            else:
                warn("Piper voice model file not found — falling back to pyttsx3")
                self.mode = "pyttsx3"

        # ── pyttsx3 Offline Fallback ─────────────────────────────────────────
        if self.mode == "pyttsx3":
            if PYTTSX3_AVAILABLE:
                try:
                    self.engine = pyttsx3.init()
                    self.engine.setProperty("rate", self.rate)
                    if self.voice_id:
                        self.engine.setProperty("voice", self.voice_id)
                    success("pyttsx3 TTS ready (offline fallback).")
                except Exception as e:
                    warn(f"pyttsx3 init failed ({e}) — falling back to system TTS")
                    self.mode = "system"
            else:
                warn("pyttsx3 not installed — falling back to system TTS")
                self.mode = "system"

        # ── Edge-TTS / System ────────────────────────────────────────────────
        if self.mode == "edge" and not EDGE_TTS_AVAILABLE:
            warn("edge-tts not installed — falling back to system TTS")
            self.mode = "system"

    def unload(self):
        """Unload TTS engine models and free RAM/CPU completely."""
        with self._lock:
            info("Unloading TTS engine models...")
            self.stop()
            self.piper_voice = None
            self.engine = None
            import gc
            gc.collect()
            success("TTS models completely unloaded (0 MB RAM/CPU consumption).")

    def reload(self):
        """Reload TTS engine model into memory."""
        with self._lock:
            info("Reloading TTS engine models...")
            self._init_engine()

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking


    def stop(self) -> None:
        """Interrupt current speech playback immediately."""
        self._stop_event.set()
        if PYGAME_AVAILABLE:
            try:
                pygame.mixer.music.stop()
            except Exception:
                pass
        if self.engine:
            try:
                self.engine.stop()
            except Exception:
                pass
        if self._sys_proc and self._sys_proc.poll() is None:
            try:
                self._sys_proc.terminate()
            except Exception:
                pass

    def speak(self, text: str) -> bool:
        """Speak the given text. Returns True if completed fully, False if interrupted."""
        if not text or not text.strip() or self.mode == "none":
            return True

        with self._lock:
            # Clean text for speech (remove markdown asterisks, hashes, etc.)
            clean_text = text.replace("*", "").replace("#", "").replace("_", "").replace("`", "")

            self._stop_event.clear()
            self._is_speaking = True
            speak_label(clean_text)

            try:
                if self.mode == "piper" and self.piper_voice:
                    return self._speak_piper(clean_text)
                elif self.mode == "pyttsx3" and self.engine:
                    return self._speak_pyttsx3(clean_text)
                elif self.mode == "edge" and EDGE_TTS_AVAILABLE:
                    return self._speak_edge(clean_text)
                elif self.mode == "system":
                    return self._speak_system(clean_text)
                else:
                    print(c(f"  [TTS disabled] Would say: {clean_text}", DIM))
                    return True
            finally:
                self._is_speaking = False

    # ── Piper Neural Synthesis ───────────────────────────────────────────────

    def _speak_piper(self, text: str) -> bool:
        """Synthesize audio with PiperVoice and play WAV file."""
        if not self.piper_voice:
            return False

        buf = io.BytesIO()
        try:
            with wave.open(buf, 'wb') as wf:
                self.piper_voice.synthesize_wav(text, wf)
        except Exception as e:
            warn(f"Piper synthesis error: {e}")
            return False

        if self._stop_event.is_set():
            return False

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(buf.getvalue())
            tmp_path = f.name

        try:
            return self._play_audio_interruptible(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    # ── pyttsx3 Synthesis ────────────────────────────────────────────────────

    def _speak_pyttsx3(self, text: str) -> bool:
        """Run pyttsx3 in thread with interrupt polling."""
        if not self.engine:
            return False

        done_event = threading.Event()
        interrupted = [False]

        def _worker():
            try:
                with self._lock:
                    self.engine.say(text)
                    self.engine.runAndWait()
            except Exception:
                pass
            finally:
                done_event.set()

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

        while not done_event.wait(timeout=0.05):
            if self._stop_event.is_set():
                interrupted[0] = True
                try:
                    self.engine.stop()
                except Exception:
                    pass
                break

        done_event.wait(timeout=1.0)
        return not interrupted[0]

    # ── Edge-TTS Synthesis ───────────────────────────────────────────────────

    def _speak_edge(self, text: str) -> bool:
        voice = self.voice_id or "en-US-JennyNeural"
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp_path = f.name

        try:
            async def _synth():
                tts = edge_tts.Communicate(text, voice)
                await tts.save(tmp_path)

            asyncio.run(_synth())
            if self._stop_event.is_set():
                return False
            return self._play_audio_interruptible(tmp_path)
        except Exception as e:
            warn(f"Edge TTS error: {e}")
            return False
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    # ── System Native TTS ────────────────────────────────────────────────────

    def _speak_system(self, text: str) -> bool:
        sys_name = _platform.system().lower()
        if sys_name == "darwin":
            cmd = ["say", text]
        elif sys_name == "windows":
            ps = f'Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak("{text}")'
            cmd = ["powershell", "-NoProfile", "-Command", ps]
        else:
            espeak = "espeak-ng" if subprocess.run(["which", "espeak-ng"], capture_output=True).returncode == 0 else "espeak"
            cmd = [espeak, "-s", str(self.rate), text]

        try:
            self._sys_proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            while self._sys_proc.poll() is None:
                if self._stop_event.is_set():
                    self._sys_proc.terminate()
                    return False
                time.sleep(0.05)
            return True
        except Exception as e:
            warn(f"System TTS error: {e}")
            return False

    # ── Interruptible Audio Player ───────────────────────────────────────────

    def _play_audio_interruptible(self, path: str) -> bool:
        """Play audio file (WAV/MP3) via PyGame or native system audio player."""
        if not os.path.exists(path):
            return False

        # Option A: PyGame Mixer (if available and functional)
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
            except Exception:
                pass  # Fall back to OS player below

        # Option B: Native System Audio Player (aplay / pw-play / paplay / afplay)
        sys_name = _platform.system().lower()
        cmd = None

        if sys_name == "darwin":
            cmd = ["afplay", path]
        elif sys_name == "windows":
            cmd = ["powershell", "-NoProfile", "-Command",
                   f"(New-Object System.Media.SoundPlayer '{path}').PlaySync()"]
        else:
            # Linux: try aplay -> pw-play -> paplay -> ffplay
            if subprocess.run(["which", "aplay"], capture_output=True).returncode == 0:
                cmd = ["aplay", "-q", path]
            elif subprocess.run(["which", "pw-play"], capture_output=True).returncode == 0:
                cmd = ["pw-play", path]
            elif subprocess.run(["which", "paplay"], capture_output=True).returncode == 0:
                cmd = ["paplay", path]
            elif subprocess.run(["which", "ffplay"], capture_output=True).returncode == 0:
                cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path]

        if not cmd:
            warn("No audio player found to play sound.")
            return False

        try:
            self._sys_proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            while self._sys_proc.poll() is None:
                if self._stop_event.is_set():
                    self._sys_proc.terminate()
                    return False
                time.sleep(0.05)
            return True
        except Exception as e:
            warn(f"Audio playback command failed ({e})")
            return False


def list_voices(mode: str = "pyttsx3") -> None:
    """Print available voices for the specified mode."""
    engine = TTSEngine(mode=mode)
    if mode == "pyttsx3" and engine.engine:
        voices = engine.engine.getProperty("voices")
        print(f"\n{BOLD}Available pyttsx3 voices:{RESET}")
        for v in voices:
            print(f"  ID:   {c(v.id, CYAN)}")
            print(f"  Name: {v.name}")
            print(f"  Lang: {v.languages}\n")
