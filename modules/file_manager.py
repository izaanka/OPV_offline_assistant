"""File manager module for OPV Voice Assistant — Create, read, delete, move, and list files."""

import os
import re
import shutil
import platform
import subprocess
import difflib
from typing import Dict, Any

import modules_registry
from indexer import query_index, query_index_smart, build_index
from modules_registry import BaseModule
from utils import warn, info, success


def resolve_case_insensitive_path(path: str) -> str:
    """Expand user path and resolve case-insensitively if exact path does not exist."""
    if not path:
        return ""
    expanded = os.path.expanduser(path.strip())
    if os.path.exists(expanded):
        return expanded

    norm_path = os.path.normpath(expanded)
    is_absolute = os.path.isabs(norm_path)
    parts = norm_path.split(os.sep)

    current = os.sep if is_absolute else "."
    for part in parts:
        if not part:
            continue
        candidate = os.path.join(current, part)
        if os.path.exists(candidate):
            current = candidate
        else:
            match_found = False
            if os.path.isdir(current):
                try:
                    for entry in os.listdir(current):
                        if entry.lower() == part.lower():
                            current = os.path.join(current, entry)
                            match_found = True
                            break
                except Exception:
                    pass
            if not match_found:
                current = candidate
    return current


def resolve_path(path: str) -> str:
    """Resolve path with ~ expansion, case-insensitivity, and home-fallback."""
    if not path:
        return ""
    resolved = resolve_case_insensitive_path(path)
    if os.path.exists(resolved):
        return resolved

    if not os.path.isabs(path):
        home_target = os.path.join("~", path)
        home_resolved = resolve_case_insensitive_path(home_target)
        if os.path.exists(home_resolved):
            return home_resolved

    return resolved


def clean_display_title(filename: str) -> str:
    """Clean filename for spoken announcement."""
    base = os.path.splitext(os.path.basename(filename))[0]
    base = re.sub(r'\(.*?\)', '', base)
    base = re.sub(r'\[.*?\]', '', base)
    base = base.replace('_', ' ')
    base = re.sub(r'\s+', ' ', base).strip()
    return base


def speak_announcement(text: str):
    """Announce text aloud prior to launching an application using configured assistant TTS voice."""
    if not text:
        return
    info(f"Speaking pre-launch announcement: {text}")
    # Try configured TTS engine callback first (e.g. Piper neural voice)
    spoken = modules_registry.speak_tts(text)
    if spoken:
        return

    # Fallback to system speech if TTS engine callback is not registered
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
            elif subprocess.run(["which", "espeak-ng"], capture_output=True).returncode == 0:
                subprocess.run(["espeak-ng", text], check=False)
            elif subprocess.run(["which", "espeak"], capture_output=True).returncode == 0:
                subprocess.run(["espeak", text], check=False)
    except Exception as e:
        warn(f"Announcement speech error: {e}")


class FileManagerModule(BaseModule):
    name = "file_manager"
    description = (
        "File management operations (read, create, delete, move, list, open, search, reindex). "
        "Use 'open' to natively open/play a file. Use 'search' to query the file index. Use 'reindex' to update the index. "
        "Parameters: {\"action\": \"read|create|delete|move|list|open|search|reindex\", \"path\": \"file_path\", "
        "\"content\": \"text content (for create)\", \"destination\": \"dest_path (for move)\"}."
    )
    requires_confirmation = False  # Dynamic per action (delete and overwrite ask for GUI confirmation)

    def can_handle_direct(self, user_input: str) -> bool:
        lower = user_input.lower().strip()
        if lower.startswith(("open ", "launch ")):
            # Only intercept simple commands like "open myxer". Complex commands go to LLM.
            if len(lower.split()) <= 4:
                return True
        
        file_triggers = [
            "read file", "open file", "cat file",
            "create file", "write file", "make file",
            "delete file", "remove file",
            "move file", "rename file",
            "list files", "show files",
            "search file", "find file", "reindex"
        ]
        return any(t in lower for t in file_triggers)

    def parse_direct_args(self, user_input: str) -> Dict[str, Any]:
        lower = user_input.lower()
        if "delete file" in lower or "remove file" in lower:
            path = user_input.split("file")[-1].strip()
            return {"action": "delete", "path": path}
        elif "read file" in lower or "cat file" in lower:
            path = user_input.split("file")[-1].strip()
            return {"action": "read", "path": path}
        elif lower.startswith(("open ", "launch ")):
            for t in ("open ", "launch "):
                if lower.startswith(t):
                    path = user_input[len(t):].strip()
                    break
            if path.startswith("file "):
                path = path[5:].strip()
            return {"action": "open", "path": path}
        elif "search file" in lower or "find file" in lower:
            trigger = "search file" if "search file" in lower else "find file"
            path = user_input.split(trigger)[-1].strip()
            return {"action": "search", "path": path}
        elif "reindex" in lower:
            return {"action": "reindex"}
        elif "list files" in lower or "show files" in lower:
            path = "."
            if " in " in lower:
                path = user_input.split(" in ")[-1].strip()
            return {"action": "list", "path": path}
        elif "create file" in lower or "write file" in lower:
            parts = user_input.split("file")[-1].strip()
            if " with content " in parts:
                p_parts = parts.split(" with content ", 1)
                return {"action": "create", "path": p_parts[0].strip(), "content": p_parts[1].strip()}
            return {"action": "create", "path": parts, "content": ""}
        elif "move file" in lower or "rename file" in lower:
            parts = user_input.split("file")[-1].strip()
            if " to " in parts:
                src, dst = parts.split(" to ", 1)
                return {"action": "move", "path": src.strip(), "destination": dst.strip()}
            return {"action": "move", "path": parts}

        return {"action": "read", "path": ""}

    def execute(self, params: Dict[str, Any], user_input: str = "") -> str:
        action = params.get("action", "").lower()
        raw_path = params.get("path", "").strip()
        content = params.get("content", "")
        raw_dest = params.get("destination", "").strip()

        path = resolve_path(raw_path) if raw_path else ""
        destination = resolve_path(raw_dest) if raw_dest else ""

        if not action:
            return "Error: File action not specified."

        # Action: REINDEX
        if action == "reindex":
            count = build_index()
            return f"Successfully rebuilt file index. Indexed {count} files."

        # Action: SEARCH
        if action == "search":
            if not raw_path:
                return "Error: No search query provided."
            matches = query_index(raw_path)
            if not matches:
                return f"No files found matching '{raw_path}'."
            return f"Found {len(matches)} matches for '{raw_path}':\n" + "\n".join(matches)

        # Action: OPEN / PLAY
        if action in ("open", "play", "launch"):
            best_score = 1.0
            search_query = os.path.basename(raw_path.rstrip("/\\")) if raw_path else ""
            
            # Check if an executable application matches the search query in the system PATH
            import shutil
            exec_path = None
            if search_query:
                exec_path = shutil.which(search_query) or shutil.which(search_query.lower())
            
            if exec_path and not (path and os.path.exists(path)):
                app_name = os.path.basename(exec_path)
                speak_announcement(f"Opening {app_name}")
                try:
                    subprocess.Popen([exec_path], start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    success(f"Launched application '{exec_path}'.")
                    return f"[SILENT_SUCCESS] Opening {app_name}."
                except Exception as e:
                    return f"Error launching application '{app_name}': {e}"
            
            if not path or not os.path.exists(path):
                matches = query_index_smart(search_query, limit=5)
                if not matches:
                    return f"Error: File '{search_query}' does not exist and could not be found."
                
                best_match_path, best_filename, best_score = matches[0]
                info(f"Top relativistic match for '{search_query}': '{best_filename}' (Score: {best_score:.2f})")
                path = best_match_path

            clean_title = clean_display_title(path)
            ext = os.path.splitext(path)[1].lower()

            audio_exts = {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac'}
            video_exts = {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv', '.wmv', '.m4v', '.3gp'}
            doc_exts   = {'.txt', '.pdf', '.docx', '.doc', '.md', '.csv', '.xlsx', '.pptx', '.rtf', '.odt', '.json', '.py', '.log'}

            if ext in audio_exts:
                verb = "Playing audio"
            elif ext in video_exts:
                verb = "Playing video"
            elif ext in doc_exts:
                verb = "Opening document"
            else:
                verb = "Opening"

            # 1. Announce BEFORE opening!
            speak_announcement(f"{verb} {clean_title}")

            # 2. Open file natively
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

        # Action: READ
        elif action == "read":
            if not path or not os.path.exists(path):
                search_query = os.path.basename(raw_path.rstrip("/\\")) if raw_path else ""
                matches = query_index_smart(search_query, limit=5)
                if matches:
                    path = matches[0][0]
                else:
                    return f"Error: File '{search_query}' does not exist."
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    data = f.read(4000)
                return f"File content of '{os.path.basename(path)}':\n{data}"
            except Exception as e:
                return f"Error reading file '{path}': {e}"

        # Action: CREATE / WRITE
        elif action in ("create", "write"):
            if not path:
                return "Error: File path not provided."

            if os.path.exists(path):
                # Request pop-up confirmation before overwriting
                confirmed = modules_registry.confirm_action(f"Overwrite existing file '{path}'?", f"New content length: {len(content)} chars")
                if not confirmed:
                    return f"Overwriting file '{path}' was CANCELED by user."

            try:
                os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                success(f"File '{path}' created successfully.")
                return f"Successfully created file '{path}' with {len(content)} bytes."
            except Exception as e:
                return f"Error creating file '{path}': {e}"

        # Action: DELETE
        elif action in ("delete", "remove"):
            if not path or not os.path.exists(path):
                return f"Error: File or path '{path}' does not exist."

            # ALWAYS request pop-up confirmation before deleting!
            confirmed = modules_registry.confirm_action(f"Delete file/directory '{path}'?", f"Target path: {os.path.abspath(path)}")
            if not confirmed:
                return f"Deletion of '{path}' was CANCELED by user."

            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
                success(f"Deleted '{path}'.")
                return f"Successfully deleted '{path}'."
            except Exception as e:
                return f"Error deleting '{path}': {e}"

        # Action: MOVE / RENAME
        elif action in ("move", "rename"):
            if not path or not os.path.exists(path):
                return f"Error: Source file '{path}' does not exist."
            if not destination:
                return "Error: Destination path not provided for move action."
            try:
                os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)
                shutil.move(path, destination)
                return f"Successfully moved '{path}' to '{destination}'."
            except Exception as e:
                return f"Error moving '{path}' to '{destination}': {e}"

        # Action: LIST
        elif action in ("list", "ls"):
            if not path:
                target_dir = "."
            elif os.path.isdir(path):
                target_dir = path
            else:
                # Try to resolve relative to home directory (e.g. "Downloads" -> "~/Downloads")
                home_path = os.path.expanduser(os.path.join("~", path))
                if os.path.isdir(home_path):
                    target_dir = home_path
                else:
                    return f"Error: Directory '{path}' does not exist."

            try:
                entries = os.listdir(target_dir)
                items = [f"{'[DIR] ' if os.path.isdir(os.path.join(target_dir, e)) else ''}{e}" for e in sorted(entries)]
                return f"Files in '{target_dir}':\n" + "\n".join(items[:50])
            except Exception as e:
                return f"Error listing directory '{target_dir}': {e}"

        return f"Unknown file action '{action}'."
