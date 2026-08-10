"""Media player module for OPV Voice Assistant — Search and play media files via LLM."""

import os
import re
import platform
import subprocess
from typing import Dict, Any

import modules_registry
from indexer import query_index_smart
from modules_registry import BaseModule
from utils import info, warn, success


# ─── Media file extensions ─────────────────────────────────────────────────────
AUDIO_EXTS = {'.mp3', '.flac', '.wav', '.aac', '.ogg', '.m4a', '.wma', '.opus', '.aiff'}
VIDEO_EXTS = {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv', '.wmv', '.m4v', '.3gp'}
MEDIA_EXTS = AUDIO_EXTS | VIDEO_EXTS


def clean_display_title(filename: str) -> str:
    """Clean filename for spoken announcement."""
    base = os.path.splitext(os.path.basename(filename))[0]
    base = re.sub(r'\(.*?\)', '', base)
    base = re.sub(r'\[.*?\]', '', base)
    base = base.replace('_', ' ')
    base = re.sub(r'\s+', ' ', base).strip()
    return base


def speak_announcement(text: str):
    """Announce text aloud using the assistant's configured TTS voice."""
    if not text:
        return
    info(f"Speaking pre-launch announcement: {text}")
    spoken = modules_registry.speak_tts(text)
    if spoken:
        return

    # Fallback to system speech
    try:
        sys_name = platform.system()
        if sys_name == "Darwin":
            subprocess.run(["say", text], check=False)
        elif sys_name == "Windows":
            ps_cmd = f"Add-Type –AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{text}');"
            subprocess.run(["powershell", "-Command", ps_cmd], check=False)
        else:
            if subprocess.run(["which", "spd-say"], capture_output=True).returncode == 0:
                subprocess.run(["spd-say", "-w", text], check=False)
    except Exception:
        pass


class MediaPlayerModule(BaseModule):
    name = "media_player"
    description = (
        "Search for and play audio/video media files by name. "
        "Uses the file index to fuzzy-match track names. "
        "Parameters: {\"track\": \"song or video name to search for\"}. "
        "Finds the best matching media file and plays it using the system default player. "
        "You calling this tool IS your permission to play — the best match opens automatically."
    )
    requires_confirmation = False

    def execute(self, params: Dict[str, Any], user_input: str = "") -> str:
        track_query = params.get("track", "").strip()

        if not track_query:
            return "Error: No 'track' name provided. Please specify a track to search for."

        # Search using the same smart indexer as file_manager
        matches = query_index_smart(track_query, limit=10)

        if not matches:
            return f"No files found matching '{track_query}'."

        # Filter to media files only
        media_matches = []
        for match_path, match_filename, match_score in matches:
            ext = os.path.splitext(match_filename)[1].lower()
            if ext in MEDIA_EXTS:
                media_matches.append((match_path, match_filename, match_score))

        if not media_matches:
            # Show what was found even though none are media
            all_names = [f"  - {fn} (score: {sc:.2f})" for _, fn, sc in matches[:5]]
            return (
                f"No audio/video files found matching '{track_query}'. "
                f"Found these non-media files instead:\n" + "\n".join(all_names)
            )

        # Pick the best match and play it
        best_path, best_filename, best_score = media_matches[0]
        return self._play_file(best_path)

    def _play_file(self, path: str) -> str:
        """Open a media file with the system's default player."""
        if not os.path.exists(path):
            return f"Error: File '{path}' does not exist."

        clean_title = clean_display_title(path)
        ext = os.path.splitext(path)[1].lower()
        verb = "Playing video" if ext in VIDEO_EXTS else "Playing"

        # Announce before opening
        speak_announcement(f"{verb} {clean_title}")

        try:
            system_name = platform.system()
            if system_name == "Darwin":
                subprocess.call(["open", path])
            elif system_name == "Windows":
                os.startfile(path)
            else:
                subprocess.call(["xdg-open", path])
            success(f"Opened '{path}' using default OS application.")
            return f"[SILENT_SUCCESS] {verb} {clean_title}."
        except Exception as e:
            return f"Error opening file '{path}': {e}"
