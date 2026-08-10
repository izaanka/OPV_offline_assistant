import os
import sys
import urllib.request
import json
from typing import Tuple

from config import PIPER_VOICE_DIR, DEFAULT_WAKE_WORD
from utils import c, info, success, warn, error, BOLD, CYAN, GREEN, YELLOW, MAGENTA, DIM, RESET

PIPER_VOICE_CATALOGUE = {
    "male": [
        {"name": "en_US-ryan-medium", "desc": "Ryan (Medium, ~60 MB) — ★ Balanced speed & naturalness [Default]", "hf_path": "en/en_US/ryan/medium", "files": ["en_US-ryan-medium.onnx", "en_US-ryan-medium.onnx.json"]},
        {"name": "en_US-ryan-high", "desc": "Ryan (Large & High Quality, ~120 MB) — Deep, studio quality male voice", "hf_path": "en/en_US/ryan/high", "files": ["en_US-ryan-high.onnx", "en_US-ryan-high.onnx.json"]},
        {"name": "en_US-danny-low", "desc": "Danny (Small & Fast, ~15 MB) — Ultra-low latency voice", "hf_path": "en/en_US/danny/low", "files": ["en_US-danny-low.onnx", "en_US-danny-low.onnx.json"]},
    ],
    "female": [
        {"name": "en_US-libritts_r-medium", "desc": "LibriTTS (Medium, ~60 MB) — ★ Natural expressive female voice [Default]", "hf_path": "en/en_US/libritts_r/medium", "files": ["en_US-libritts_r-medium.onnx", "en_US-libritts_r-medium.onnx.json"]},
        {"name": "en_US-amy-medium", "desc": "Amy (Medium, ~60 MB) — Balanced speed & high naturalness", "hf_path": "en/en_US/amy/medium", "files": ["en_US-amy-medium.onnx", "en_US-amy-medium.onnx.json"]},
        {"name": "en_US-lessac-high", "desc": "Lessac (Large & High Quality, ~120 MB) — Clear, studio quality female voice", "hf_path": "en/en_US/lessac/high", "files": ["en_US-lessac-high.onnx", "en_US-lessac-high.onnx.json"]},
        {"name": "en_US-kathleen-low", "desc": "Kathleen (Small & Fast, ~15 MB) — Ultra-low latency voice", "hf_path": "en/en_US/kathleen/low", "files": ["en_US-kathleen-low.onnx", "en_US-kathleen-low.onnx.json"]},
    ],
}

_MALE_KEYWORDS = {"david", "mark", "daniel", "james", "michael", "george", "thomas", "richard", "male", "man", "guy", "ryan", "danny"}
_FEMALE_KEYWORDS = {"zira", "hazel", "victoria", "samantha", "lisa", "karen", "susan", "amy", "alice", "emma", "female", "woman", "girl", "kathleen", "jenny", "lessac", "libritts"}

def _progress_hook(count, block_size, total_size):
    """Print clean visual progress bar for urllib downloads."""
    if total_size > 0:
        percent = min(100, int(count * block_size * 100 / total_size))
        bar_len = 30
        filled = int(bar_len * percent / 100)
        bar = "=" * filled + ">" if filled < bar_len else "=" * bar_len
        bar = bar.ljust(bar_len)
        mb_downloaded = (count * block_size) / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        print(f"\r  Progress: [{c(bar, CYAN)}] {percent:3d}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="", flush=True)

def _download_piper_voice(voice_info: dict) -> str:
    """Downloads a piper voice model from HuggingFace with progress bar."""
    os.makedirs(PIPER_VOICE_DIR, exist_ok=True)
    base_url = f"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/{voice_info['hf_path']}/"
    onnx_path = ""
    for file in voice_info["files"]:
        target_path = os.path.join(PIPER_VOICE_DIR, file)
        if file.endswith(".onnx"):
            onnx_path = target_path
        if not os.path.exists(target_path):
            info(f"Downloading Piper voice file: {file}...")
            url = base_url + file
            try:
                urllib.request.urlretrieve(url, target_path, _progress_hook)
                print()  # New line after progress bar
                success(f"Downloaded {file}")
            except Exception as e:
                print()
                error(f"Download failed for {file}: {e}")
                if os.path.exists(target_path):
                    os.remove(target_path)
                sys.exit(1)
    return onnx_path

def select_tts_voice(args) -> Tuple[str, str, str]:
    """Shows voice model picker. Returns (piper_voice_path, voice_id, tts_mode)."""
    tts_mode = getattr(args, 'tts', 'piper')
    piper_voice_path = getattr(args, 'piper_voice', None)
    
    if piper_voice_path and os.path.isfile(piper_voice_path):
        return piper_voice_path, None, tts_mode

    if tts_mode == 'piper':
        gender_flag = getattr(args, 'gender', None)
        if not gender_flag:
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
                raw = "2"
            gender = "male" if raw == "1" else "female"
        else:
            gender = gender_flag.lower()

        voices = PIPER_VOICE_CATALOGUE.get(gender, PIPER_VOICE_CATALOGUE["female"])
        print(f"\n{BOLD}{CYAN}Select a {gender.capitalize()} Piper voice model:{RESET}")
        print(c("─" * 65, DIM))
        default_idx = 1
        for i, v in enumerate(voices, 1):
            onnx_path = os.path.join(PIPER_VOICE_DIR, v["files"][0])
            downloaded = os.path.isfile(onnx_path)
            status = c("[Ready]", GREEN) if downloaded else c("[Download required]", YELLOW)
            if "libritts" in v["name"].lower():
                default_idx = i
            print(f"  {c(f'[{i}]', CYAN)} {v['desc']}  {status}")
        print(c("─" * 65, DIM))
        print(f"  {c('[Enter]', DIM)} use default ({default_idx})")
        print()

        try:
            raw2 = input(f"{BOLD}Your choice (1-{len(voices)}): {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            raw2 = ""

        idx = (int(raw2) - 1) if raw2.isdigit() and 1 <= int(raw2) <= len(voices) else (default_idx - 1)
        voice_info = voices[idx]
        piper_path = _download_piper_voice(voice_info)
        return piper_path, voice_info['name'], 'piper'
    
    return "", "", tts_mode

def select_wake_word_and_name(args, config: dict) -> Tuple[str, str]:
    """Interactive wake word and assistant name prompt."""
    if getattr(args, 'reconfigure', False) or 'wake_word' not in config:
        print(f"\n{BOLD}{CYAN}Select Assistant Name & Wake Word:{RESET}")
        print(c("─" * 60, DIM))
        print("  Set the name and wake word to call your assistant")
        print(f"  Examples: {c('hey', GREEN)}, {c('stellar', GREEN)}, {c('jarvis', GREEN)}, {c('alexa', GREEN)}")
        print(c("─" * 60, DIM))
        default_val = config.get("wake_word", DEFAULT_WAKE_WORD)
        print(f"  {c('[Enter]', DIM)} use default (\"{default_val}\")")
        print()
        try:
            raw = input(f"{BOLD}Assistant Name / Wake Word [{default_val}]: {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            raw = ""
        if raw:
            wake_word = raw.lower()
            name = raw.capitalize()
        else:
            wake_word = default_val.lower()
            name = config.get("assistant_name", "Stellar" if wake_word == "stellar" else "Assistant")
        return wake_word, name
    return config.get('wake_word', DEFAULT_WAKE_WORD), config.get('assistant_name', 'Stellar')
