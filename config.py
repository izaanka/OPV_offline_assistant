"""Configuration persistence and CLI argument parsing for OPV Voice Assistant."""

import argparse
import json
import os
from typing import Dict, Any

from utils import info, success, warn, error

# ─── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_MODEL      = "llama3.1:8b"
DEFAULT_WAKE_WORD  = "hey"
DEFAULT_TTS        = "piper"
PIPER_VOICE_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "piper-voices")
LISTEN_TIMEOUT     = 8
PHRASE_LIMIT       = 15
AMBIENT_DURATION   = 1.0
CONVERSATION_HIST  = 10

# ─── Configuration File ───────────────────────────────────────────────────────
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".config")


def load_config() -> Dict[str, Any]:
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


def save_config(cfg: Dict[str, Any]) -> None:
    """Save configuration to .config file."""
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        success("Saved active configuration to .config")
    except Exception as e:
        warn(f"Could not save configuration to .config ({e})")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the assistant."""
    parser = argparse.ArgumentParser(
        description="Local AI Voice Assistant powered by Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python assistant.py
  python assistant.py --model mistral
  python assistant.py --stt whisper --whisper-model small.en
  python assistant.py --wake-word stellar --rate 160
  python assistant.py --reconfigure
  python assistant.py --list-voices
        """
    )
    parser.add_argument("--model",          default=None,
                        help="Ollama model to use (omit to pick interactively)")
    parser.add_argument("--wake-word",      default=DEFAULT_WAKE_WORD,
                        help=f"Wake word to listen for (default: {DEFAULT_WAKE_WORD})")
    parser.add_argument("--assistant-name", default=None,
                        help="Name of the assistant (default: Assistant or wake word)")
    parser.add_argument("--stt",            choices=["whisper", "vosk"], default="whisper",
                        help="STT engine: whisper (BPE/GPU) or vosk (dictionary, fallback)")
    parser.add_argument("--whisper-model",  default="small.en",
                        choices=["tiny", "tiny.en", "base", "base.en",
                                 "small", "small.en", "medium", "medium.en",
                                 "large-v3", "large-v3-turbo",
                                 "distil-large-v3", "distil-medium.en", "distil-small.en"],
                        help="Whisper model size (default: small.en)")
    parser.add_argument("--no-rescore",     action="store_true",
                        help="Disable LLM rescoring pass")
    parser.add_argument("--tts",            choices=["piper", "pyttsx3", "edge", "none"],
                        default=DEFAULT_TTS,
                        help="TTS engine (default: piper)")
    parser.add_argument("--gender",         choices=["male", "female"], default=None,
                        help="Voice gender to use")
    parser.add_argument("--voice",          default=None,
                        help="Exact voice ID for pyttsx3 or edge-tts")
    parser.add_argument("--edge-voice",     default=None,
                        help="edge-tts voice name")
    parser.add_argument("--piper-voice",    default=None, metavar="PATH",
                        help="Path to a Piper .onnx voice file")
    parser.add_argument("--rate",           type=int, default=175,
                        help="Speech rate for pyttsx3 (default: 175)")
    parser.add_argument("--list-voices",    action="store_true",
                        help="List available TTS voices and exit")
    parser.add_argument("--vosk-model",     default=None, metavar="PATH",
                        help="Path to a Vosk model directory")
    parser.add_argument("--reconfigure",    action="store_true",
                        help="Reset saved options and re-select interactively")
    return parser.parse_args()
