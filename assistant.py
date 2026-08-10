#!/usr/bin/env python3
"""
Stellar — Local AI Voice Assistant
───────────────────────────────────────
STT:  faster-whisper (BPE subword, GPU) or Vosk (dictionary, CPU fallback)
LLM:  Ollama (any local model)
TTS:  Piper (neural, offline) / pyttsx3 / edge-tts / system native
"""

import os
import sys
import time
import threading
from typing import Optional
from typing import Optional, Dict, Any

from utils import (c, info, success, warn, error, speak_label, user_label, ai_label,
                   RESET, BOLD, CYAN, GREEN, YELLOW, RED, MAGENTA, DIM)
from config import (parse_args, load_config, save_config,
                    DEFAULT_MODEL, DEFAULT_WAKE_WORD, DEFAULT_TTS)
from memory import load_memory, save_memory, add_memory_fact, auto_extract_facts
from normalizer import save_alias
from llm import OllamaLLM, select_ollama_model, list_ollama_models
from stt import SpeechListener, select_stt_engine
from tts import TTSEngine
from voice_setup import select_tts_voice, select_wake_word_and_name
from rescorer import build_context_bias
from chat_gui import ChatGUI
import modules_registry
import indexer


class VoiceAssistant:
    """Main voice assistant loop — wake word → listen → think → speak + Desktop Chat GUI."""

    def __init__(self, model: str, tts_mode: str, voice_id: Optional[str], tts_rate: int,
                 wake_word: str, assistant_name: str = "Assistant",
                 stt_engine: str = "whisper", whisper_model: str = "small.en",
                 vosk_model_path: Optional[str] = None,
                 piper_voice_path: Optional[str] = None,
                 enable_rescoring: bool = True,
                 engine: str = "opv"):
        self.wake_word = wake_word.lower()
        self.assistant_name = assistant_name.strip()
        self.engine = engine.lower()
        modules_registry.load_modules()
        # Automatically update file index asynchronously on bootup
        threading.Thread(target=indexer.build_index, daemon=True).start()
        
        if self.engine == "openclaw":
            try:
                from openclaw_engine import OpenClawLLM
                self.llm = OpenClawLLM(model=model, assistant_name=self.assistant_name)
            except ImportError:
                print("[WARN] openclaw_engine not found, falling back to opv engine.")
                from llm import OllamaLLM
                self.llm = OllamaLLM(model=model, assistant_name=self.assistant_name)
                self.engine = "opv"
        else:
            from llm import OllamaLLM
            self.llm = OllamaLLM(model=model, assistant_name=self.assistant_name)

        self.tts = TTSEngine(mode=tts_mode, voice_id=voice_id, rate=tts_rate,
                             piper_voice_path=piper_voice_path)
        modules_registry.register_tts_callback(self.tts.speak)
        self.listener = SpeechListener(stt_engine=stt_engine, whisper_model=whisper_model,
                                       vosk_model_path=vosk_model_path,
                                       enable_rescoring=enable_rescoring)
        self.gui = ChatGUI(
            assistant_name=self.assistant_name,
            model_name=self.llm.model,
            wake_word=self.wake_word,
            stt_engine=self.listener.stt_engine,
            whisper_model=self.listener.whisper_model_name,
            tts_engine=self.tts.mode,
            piper_voice=self.tts.piper_voice_path or "",
            on_send_message=self._handle_gui_text_input,
            on_toggle_stt=self._handle_toggle_stt,
            on_toggle_tts=self._handle_toggle_tts,
            on_stop_speech=self._handle_stop_speech,
            on_save_defaults=self._handle_save_defaults,
            get_ollama_models=list_ollama_models
        )
        self._running = False

    def _handle_toggle_stt(self, enabled: bool):
        """Kills or reinstates STT engine, unloading RAM/VRAM models when killed."""
        if enabled:
            self.listener.reload_models()
        else:
            self.listener.unload_models()

    def _handle_toggle_tts(self, enabled: bool):
        """Kills or reinstates TTS engine, unloading models when killed."""
        if enabled:
            self.tts.reload()
        else:
            self.tts.unload()

    def _handle_stop_speech(self):
        """Interrupts current speech playback."""
        self.tts.stop()

    def _handle_save_defaults(self, new_config: Dict[str, Any]):
        """Save updated configuration to .config and apply live to active assistant."""
        save_config(new_config)
        self.assistant_name = new_config.get("assistant_name", self.assistant_name)
        self.wake_word = new_config.get("wake_word", self.wake_word).lower()
        self.llm.model = new_config.get("model", self.llm.model)
        self.llm.assistant_name = self.assistant_name

        stt_changed = (new_config.get("stt_engine") != self.listener.stt_engine or
                       new_config.get("whisper_model") != self.listener.whisper_model_name)
        tts_changed = (new_config.get("tts") != self.tts.mode or
                       new_config.get("piper_voice_path") != self.tts.piper_voice_path)

        if stt_changed:
            self.listener.stt_engine = new_config.get("stt_engine", self.listener.stt_engine)
            self.listener.whisper_model_name = new_config.get("whisper_model", self.listener.whisper_model_name)
            if self.gui and self.gui.stt_enabled:
                info("Reloading STT engine with new model settings...")
                self.listener.reload_models()

        if tts_changed:
            self.tts.mode = new_config.get("tts", self.tts.mode)
            self.tts.piper_voice_path = new_config.get("piper_voice_path", self.tts.piper_voice_path)
            if self.gui and self.gui.tts_enabled:
                info("Reloading TTS engine with new settings...")
                self.tts.reload()

        info(f"Updated live configuration: Name='{self.assistant_name}', Model='{self.llm.model}', Wake='{self.wake_word}', STT='{self.listener.stt_engine}', TTS='{self.tts.mode}'")

    def _handle_gui_text_input(self, text: str):
        """Callback when user types and sends a message in the Chat GUI."""
        self.tts.stop()  # Stop any ongoing speech when user sends a new message
        if self.gui:
            self.gui.update_status("● Thinking...", "#89b4fa")
        self._process_and_respond(text)
        if self.gui:
            self.gui.update_status("● Listening...", "#a6e3a1")

    def _print_banner(self):
        stt_label = f"Whisper ({self.listener.stt_engine})" if self.listener._whisper_model \
                    else "Vosk (dictionary fallback)"
        banner = f"""
{BOLD}{CYAN}=== OPV Voice Assistant Initialization ==={RESET}
  Name     : {c(self.assistant_name, GREEN)}
  Model    : {c(self.llm.model, GREEN)}
  STT      : {c(stt_label, GREEN)}
  Wake word: {c(f'"{self.wake_word}"', YELLOW)}
  TTS      : {c(self.tts.mode, MAGENTA)}
  GUI Chat : {c("Active (External Window)", GREEN)}
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

        # Warm up LLM into GPU/RAM
        self.llm.warmup()

        # Calibrate mic
        self.listener.calibrate()

        # Set contextual bias for Whisper
        if self.listener._whisper_model:
            bias = build_context_bias(self.wake_word, self.assistant_name)
            self.listener.set_context_bias(bias)
            info(f"Contextual bias loaded ({len(bias)} chars)")

        self.tts.speak(f"Hello! Say {self.wake_word} to wake me up.")
        print(f"\n{BOLD}Listening for wake word: \"{self.wake_word}\" ...{RESET}\n")

        self._running = True

        # Run voice listening loop in background thread
        voice_thread = threading.Thread(target=self._voice_loop, daemon=True)
        voice_thread.start()

        # Launch Desktop GUI window on main thread
        try:
            info("Opening external desktop chat window...")
            self.gui.start_gui()
        except (KeyboardInterrupt, SystemExit):
            info("Interrupt received. Shutting down assistant...")
            self._running = False
            sys.exit(0)
        except Exception as e:
            warn(f"GUI closed or display unavailable: {e}")
            voice_thread.join()

    def _voice_loop(self):
        """Background loop continuously listening for wake word."""
        try:
            while self._running:
                self._wait_for_wake_word()
        except KeyboardInterrupt:
            print(f"\n{DIM}Interrupted by user.{RESET}")
        finally:
            self._running = False

    def _wait_for_wake_word(self):
        """Continuously listen until wake word is detected."""
        if self.gui and not self.gui.stt_enabled:
            time.sleep(0.5)
            return

        text = self.listener.listen_once()
        if text is None:
            return

        print(f"{DIM}heard: {text}{RESET}")

        if self.listener.contains_wake_word(text, self.wake_word):
            # Extract any query that came right after wake word
            words = text.lower().split()
            idx = next((i for i, w in enumerate(words)
                        if w == self.wake_word or
                        (len(w) > 3 and __import__('difflib').SequenceMatcher(
                            None, w, self.wake_word).ratio() >= 0.75)), -1)
            inline_query = " ".join(words[idx + 1:]).strip() if idx >= 0 else ""

            print(f"\n{BOLD}{GREEN}⚡ Wake word detected!{RESET}")
            if self.gui:
                self.gui.update_status("⚡ Wake word detected!", "#f9e2af")
            self.tts.speak("Yes?")

            self._handle_query(inline_query)


    def _handle_query(self, prefill: str = ""):
        """Capture command and enter 30-second conversational follow-up window."""
        if self.gui and not self.gui.stt_enabled:
            return

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

        # ── 30-Second Conversational Follow-Up Window ──────────────────────
        FOLLOWUP_WINDOW = 30.0
        followup_start = time.time()

        while (time.time() - followup_start < FOLLOWUP_WINDOW) and self._running:
            if self.gui and not self.gui.stt_enabled:
                break

            remaining = int(FOLLOWUP_WINDOW - (time.time() - followup_start))
            print(f"\r\033[K{c(f'💬 Conversational follow-up active ({remaining}s remaining — speak directly)...', CYAN)}",
                  end="", flush=True)

            text = self.listener.listen_once_timeout(timeout=4.0)
            if not text or not text.strip():
                continue

            text = text.strip()
            print("\r\033[K", end="", flush=True)

            # Strip wake word if repeated in follow-up mode
            words = text.lower().split()
            if self.wake_word in words:
                idx = words.index(self.wake_word)
                text = " ".join(words[idx + 1:]).strip()
                if not text:
                    self.tts.speak("Yes?")
                    followup_start = time.time()
                    continue

            user_label(text)
            should_continue = self._process_and_respond(text)
            if should_continue:
                followup_start = time.time()
            else:
                break

        print(f"\r\033[K\n{BOLD}Listening for wake word: \"{self.wake_word}\" ...{RESET}\n")

    def _process_and_respond(self, user_input: str) -> bool:
        """Process user input, handle special commands, auto-extract facts, and chat.
        Returns False if exiting."""
        cmd = user_input.lower().strip()

        # Auto-extract facts from speech
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

        # Alias command
        if cmd.startswith("alias ") and " as " in cmd:
            parts = user_input[6:].split(" as ", 1)
            if len(parts) == 2:
                phrase, correction = parts[0].strip(), parts[1].strip()
                if phrase and correction:
                    save_alias(phrase, correction)
                    success(f"Saved alias: '{phrase}' → '{correction}'")
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

        # Post user query to GUI if available
        if self.gui:
            self.gui.update_status("● Thinking...", "#89b4fa")

        # Query LLM
        info("Thinking...")
        def _on_think_callback(think_text: str):
            if self.gui:
                self.gui.post_think_message(think_text)

        response = self.llm.chat(user_input, on_think=_on_think_callback)
        ai_label(response)

        if self.gui:
            self.gui.post_ai_message(response)
            self.gui.update_status("● Speaking...", "#f9e2af")

        # Speak response (interruptible with "stop")
        if "[SILENT_SUCCESS]" not in response:
            self._speak_with_stop_listener(response)
        else:
            info("Skipping post-playback TTS output (music playing).")

        if self.gui:
            self.gui.update_status("● Listening..." if (self.gui and self.gui.stt_enabled) else "● STT Muted", "#a6e3a1" if (self.gui and self.gui.stt_enabled) else "#f38ba8")
        return True


    def _speak_with_stop_listener(self, text: str):
        """Speak text while a background thread listens for 'stop'."""
        if self.gui and not self.gui.tts_enabled:
            info("TTS is killed/disabled — skipping audio speech output.")
            return

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

        listener_thread = threading.Thread(target=_stop_listener, daemon=True)
        listener_thread.start()

        self.tts.speak(text)

        speech_done.set()
        listener_thread.join(timeout=2.0)

        if stopped_by_user[0]:
            print(c("[✋] Response stopped by user.", YELLOW))


# ─── Entry Point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    config = load_config() if not args.reconfigure else {}

    # Restore saved options if CLI flags were omitted
    if not args.model and config.get("model"):
        args.model = config["model"]
        info(f"Loaded saved LLM model from .config: {args.model}")

    if not args.piper_voice and config.get("piper_voice"):
        if os.path.exists(config["piper_voice"]):
            args.piper_voice = config["piper_voice"]
            info(f"Loaded saved Piper voice from .config: {os.path.basename(args.piper_voice)}")

    if not args.gender and config.get("gender"):
        args.gender = config["gender"]
        info(f"Loaded saved voice gender from .config: {args.gender}")

    # ── Wake Word & Assistant Name selection ──────────────────────────────
    chosen_wake_word, chosen_assistant_name = select_wake_word_and_name(args, config)

    # ── Model selection ──────────────────────────────────────────────────
    chosen_llm_model = select_ollama_model(args.model)

    # ── STT Engine selection ─────────────────────────────────────────────
    chosen_stt, chosen_whisper_model, chosen_vosk_path = select_stt_engine(args, config)

    # ── Voice selection ──────────────────────────────────────────────────
    piper_voice_path, voice_id, chosen_tts = select_tts_voice(args)

    # Fallback: if Piper mode but no voice resolved
    if chosen_tts == "piper" and not piper_voice_path:
        warn("No Piper voice resolved — falling back to pyttsx3.")
        chosen_tts = "pyttsx3"

    # Save active configuration
    new_config = {
        "model": chosen_llm_model,
        "tts": chosen_tts,
        "voice_id": voice_id,
        "wake_word": chosen_wake_word,
        "assistant_name": chosen_assistant_name,
        "stt": chosen_stt,
        "whisper_model": chosen_whisper_model,
        "vosk_model_path": chosen_vosk_path,
        "piper_voice_path": piper_voice_path,
        "engine": chosen_engine
    }
    save_config(new_config)

    assistant = VoiceAssistant(
        model=chosen_llm_model,
        tts_mode=chosen_tts,
        voice_id=voice_id,
        tts_rate=args.rate,
        wake_word=chosen_wake_word,
        assistant_name=chosen_assistant_name,
        stt_engine=chosen_stt,
        whisper_model=chosen_whisper_model,
        vosk_model_path=chosen_vosk_path,
        piper_voice_path=piper_voice_path,
        enable_rescoring=not args.no_rescore,
        engine=chosen_engine
    )

    if args.list_voices:
        assistant.tts.list_voices()
        sys.exit(0)

    assistant.run()


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, SystemExit):
        print("\n[INFO] Assistant process terminated cleanly.")
        sys.exit(0)
