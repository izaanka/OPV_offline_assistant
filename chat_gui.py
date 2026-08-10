"""Desktop Chat GUI for OPV Voice Assistant using Tkinter — Professional Layout."""

import queue
import threading
from typing import Callable, Optional, Dict, Any, List

from utils import info, warn, success

try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext, messagebox
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False
    tk = None


class ChatGUI:
    """Tkinter Desktop Chat GUI Window for OPV Voice Assistant."""

    def __init__(
        self,
        assistant_name: str = "Assistant",
        model_name: str = "llama3.1:8b",
        wake_word: str = "hey",
        stt_engine: str = "whisper",
        whisper_model: str = "small.en",
        tts_engine: str = "piper",
        piper_voice: str = "",
        engine: str = "opv",
        on_send_message: Optional[Callable[[str], None]] = None,
        on_toggle_stt: Optional[Callable[[bool], None]] = None,
        on_toggle_tts: Optional[Callable[[bool], None]] = None,
        on_stop_speech: Optional[Callable[[], None]] = None,
        on_save_defaults: Optional[Callable[[Dict[str, Any]], None]] = None,
        get_ollama_models: Optional[Callable[[], List[str]]] = None
    ):
        self.assistant_name = assistant_name
        self.model_name = model_name
        self.wake_word = wake_word
        self.stt_engine = stt_engine
        self.whisper_model = whisper_model
        self.tts_engine = tts_engine
        self.piper_voice = piper_voice
        self.engine = engine
        self.on_send_message = on_send_message
        self.on_toggle_stt = on_toggle_stt
        self.on_toggle_tts = on_toggle_tts
        self.on_stop_speech = on_stop_speech
        self.on_save_defaults = on_save_defaults
        self.get_ollama_models = get_ollama_models

        self.stt_enabled = True
        self.tts_enabled = True

        self.msg_queue = queue.Queue()
        self.root: Optional[tk.Tk] = None
        self.title_label: Optional[tk.Label] = None
        self.status_label: Optional[tk.Label] = None
        self.chat_display: Optional[scrolledtext.ScrolledText] = None
        self._is_closed = False

    def start_gui(self):
        """Build and launch the Tkinter GUI window on current thread."""
        if not TKINTER_AVAILABLE:
            warn("Tkinter is not installed on system. Pop-up chat GUI skipped.")
            return

        self.root = tk.Tk()
        self.root.title(f"{self.assistant_name} — Voice & Chat Assistant")
        self.root.geometry("680x750")
        self.root.minsize(520, 550)
        self.root.configure(bg="#181825")

        # ── Configure Grid Layout on Root Window (Row 1 Expands Flexibly) ─────
        self.root.grid_rowconfigure(0, weight=0)  # Top Header & Toolbar
        self.root.grid_rowconfigure(1, weight=1)  # Main Chat Display (EXPANDS)
        self.root.grid_rowconfigure(2, weight=0)  # Bottom Input Bar (FIXED)
        self.root.grid_columnconfigure(0, weight=1)

        # ── Top Header Frame (Row 0) ─────────────────────────────────────────
        header = tk.Frame(self.root, bg="#11111b")
        header.grid(row=0, column=0, sticky="ew", padx=0, pady=0)
        header.columnconfigure(0, weight=1)

        # Title Sub-frame
        title_frame = tk.Frame(header, bg="#11111b")
        title_frame.pack(side=tk.LEFT, fill=tk.Y, padx=16, pady=10)

        self.title_label = tk.Label(
            title_frame,
            text=f"{self.assistant_name}",
            font=("Segoe UI", 13, "bold"),
            fg="#cdd6f4",
            bg="#11111b"
        )
        self.title_label.pack(side=tk.TOP, anchor="w")

        self.status_label = tk.Label(
            title_frame,
            text="Ready",
            font=("Segoe UI", 9),
            fg="#a6e3a1",
            bg="#11111b"
        )
        self.status_label.pack(side=tk.TOP, anchor="w")

        # Toolbar Control Buttons (Right aligned in Header)
        toolbar = tk.Frame(header, bg="#11111b")
        toolbar.pack(side=tk.RIGHT, padx=16, pady=10)

        # STT Toggle Button
        self.stt_btn = tk.Button(
            toolbar,
            text="STT: ON",
            font=("Segoe UI", 9, "bold"),
            bg="#313244",
            fg="#a6e3a1",
            activebackground="#45475a",
            activeforeground="#a6e3a1",
            relief=tk.FLAT,
            cursor="hand2",
            padx=10,
            pady=4,
            command=self._on_stt_toggle_click
        )
        self.stt_btn.pack(side=tk.LEFT, padx=3)

        # TTS Toggle Button
        self.tts_btn = tk.Button(
            toolbar,
            text="TTS: ON",
            font=("Segoe UI", 9, "bold"),
            bg="#313244",
            fg="#a6e3a1",
            activebackground="#45475a",
            activeforeground="#a6e3a1",
            relief=tk.FLAT,
            cursor="hand2",
            padx=10,
            pady=4,
            command=self._on_tts_toggle_click
        )
        self.tts_btn.pack(side=tk.LEFT, padx=3)

        # Stop Speech Button
        stop_btn = tk.Button(
            toolbar,
            text="Stop Speech",
            font=("Segoe UI", 9, "bold"),
            bg="#f38ba8",
            fg="#11111b",
            activebackground="#f9a8d4",
            activeforeground="#11111b",
            relief=tk.FLAT,
            cursor="hand2",
            padx=10,
            pady=4,
            command=self._on_stop_speech_click
        )
        stop_btn.pack(side=tk.LEFT, padx=3)

        settings_btn = tk.Button(
            toolbar,
            text="Settings",
            font=("Segoe UI", 9, "bold"),
            bg="#89b4fa",
            fg="#11111b",
            activebackground="#b4befe",
            activeforeground="#11111b",
            relief=tk.FLAT,
            cursor="hand2",
            padx=10,
            pady=4,
            command=self._open_settings_dialog
        )
        settings_btn.pack(side=tk.LEFT, padx=3)

        # Memory Editor Button
        memory_btn = tk.Button(
            toolbar,
            text="Memory",
            font=("Segoe UI", 9, "bold"),
            bg="#f9e2af",
            fg="#11111b",
            activebackground="#fae3b0",
            activeforeground="#11111b",
            relief=tk.FLAT,
            cursor="hand2",
            padx=10,
            pady=4,
            command=self._open_memory_editor
        )
        memory_btn.pack(side=tk.LEFT, padx=3)

        # ── Main Chat Log Frame (Row 1) ─────────────────────────────────────
        chat_frame = tk.Frame(self.root, bg="#181825")
        chat_frame.grid(row=1, column=0, sticky="nsew", padx=14, pady=(8, 4))
        chat_frame.rowconfigure(0, weight=1)
        chat_frame.columnconfigure(0, weight=1)

        self.chat_display = scrolledtext.ScrolledText(
            chat_frame,
            wrap=tk.WORD,
            font=("Segoe UI", 11),
            bg="#1e1e2e",
            fg="#cdd6f4",
            insertbackground="#cdd6f4",
            selectbackground="#45475a",
            relief=tk.FLAT,
            padx=14,
            pady=14
        )
        self.chat_display.grid(row=0, column=0, sticky="nsew")

        # Configure tags for clean, readable message log
        self.chat_display.tag_configure("user_tag", foreground="#89b4fa", font=("Segoe UI", 11, "bold"))
        self.chat_display.tag_configure("ai_tag", foreground="#a6e3a1", font=("Segoe UI", 11, "bold"))
        self.chat_display.tag_configure("system_tag", foreground="#f9e2af", font=("Segoe UI", 10, "italic"))
        self.chat_display.tag_configure("think_tag", foreground="#89dceb", font=("Segoe UI", 10, "italic"))
        self.chat_display.tag_configure("msg_text", foreground="#cdd6f4", font=("Segoe UI", 11))

        # ── Bottom Input Bar (Row 2 - Fixed at Bottom) ──────────────────────
        input_frame = tk.Frame(self.root, bg="#181825")
        input_frame.grid(row=2, column=0, sticky="ew", padx=14, pady=(4, 14))
        input_frame.columnconfigure(0, weight=1)

        self.entry_var = tk.StringVar()
        self.entry = tk.Entry(
            input_frame,
            textvariable=self.entry_var,
            font=("Segoe UI", 11),
            bg="#313244",
            fg="#cdd6f4",
            insertbackground="#cdd6f4",
            relief=tk.FLAT,
            bd=5
        )
        self.entry.grid(row=0, column=0, sticky="ew", padx=(0, 8), ipady=6)
        self.entry.bind("<Return>", self._on_send_click)

        send_btn = tk.Button(
            input_frame,
            text="Send",
            font=("Segoe UI", 11, "bold"),
            bg="#89b4fa",
            fg="#11111b",
            activebackground="#b4befe",
            activeforeground="#11111b",
            relief=tk.FLAT,
            cursor="hand2",
            padx=16,
            pady=4,
            command=self._on_send_click
        )
        send_btn.grid(row=0, column=1, sticky="e")

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Welcome message in chat log
        self._append_message("System", f"Welcome to {self.assistant_name}. Speak aloud or type your message below.", "system_tag")

        # Periodically check message queue from background threads
        self.root.after(100, self._process_queue)
        try:
            self.root.mainloop()
        except (KeyboardInterrupt, SystemExit):
            try:
                self.root.destroy()
            except Exception:
                pass

    def _on_stt_toggle_click(self):
        self.stt_enabled = not self.stt_enabled
        if self.stt_enabled:
            self.stt_btn.config(text="STT: ON", fg="#a6e3a1")
            self._append_message("System", "STT reinstated. Speech recognition active.", "system_tag")
        else:
            self.stt_btn.config(text="STT: OFF", fg="#f38ba8")
            self._append_message("System", "STT disabled. Model unloaded from memory.", "system_tag")

        if self.on_toggle_stt:
            threading.Thread(target=self.on_toggle_stt, args=(self.stt_enabled,), daemon=True).start()

    def _on_tts_toggle_click(self):
        self.tts_enabled = not self.tts_enabled
        if self.tts_enabled:
            self.tts_btn.config(text="TTS: ON", fg="#a6e3a1")
            self._append_message("System", "TTS reinstated. Speech audio output active.", "system_tag")
        else:
            self.tts_btn.config(text="TTS: OFF", fg="#f38ba8")
            self._append_message("System", "TTS disabled. Audio engine unloaded from memory.", "system_tag")

        if self.on_toggle_tts:
            threading.Thread(target=self.on_toggle_tts, args=(self.tts_enabled,), daemon=True).start()

    def _on_stop_speech_click(self):
        if self.on_stop_speech:
            self._append_message("System", "Speech output stopped.", "system_tag")
            threading.Thread(target=self.on_stop_speech, daemon=True).start()

    def _open_settings_dialog(self):
        dlg = tk.Toplevel(self.root)
        dlg.title("Assistant Configuration")
        dlg.geometry("520x450")
        dlg.configure(bg="#1e1e2e")
        dlg.transient(self.root)
        dlg.grab_set()

        # Title Label
        title_lbl = tk.Label(dlg, text="Assistant Configuration", font=("Segoe UI", 13, "bold"), fg="#89b4fa", bg="#1e1e2e")
        title_lbl.pack(pady=(15, 10))

        # Main frame for two columns
        form_frame = tk.Frame(dlg, bg="#1e1e2e")
        form_frame.pack(fill=tk.BOTH, expand=True, padx=20)
        
        form_frame.columnconfigure(1, weight=1)

        # Helper to create rows
        vars_dict = {}
        row = 0

        def add_field(label_text, var_name, init_val, options=None):
            nonlocal row
            lbl = tk.Label(form_frame, text=label_text, font=("Segoe UI", 10, "bold"), fg="#cdd6f4", bg="#1e1e2e", anchor="w")
            lbl.grid(row=row, column=0, sticky="w", pady=(0, 10), padx=(0, 10))
            
            var = tk.StringVar(value=init_val)
            vars_dict[var_name] = var

            if options:
                cb = ttk.Combobox(form_frame, textvariable=var, values=options, font=("Segoe UI", 10), state="normal" if var_name == "model_var" else "readonly")
                cb.grid(row=row, column=1, sticky="ew", pady=(0, 10))
            else:
                ent = tk.Entry(form_frame, textvariable=var, font=("Segoe UI", 10), bg="#313244", fg="#cdd6f4", insertbackground="#cdd6f4", relief=tk.FLAT)
                ent.grid(row=row, column=1, sticky="ew", pady=(0, 10), ipady=3)
            row += 1

        current_whisper = getattr(self, "whisper_model", "small.en")
        current_piper = getattr(self, "piper_voice", "")
        current_stt = getattr(self, "stt_engine", "whisper")
        current_tts = getattr(self, "tts_engine", "piper")

        ollama_models = self.get_ollama_models() if self.get_ollama_models else [self.model_name]
        whisper_models = ["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en", "large-v3", "large-v3-turbo", "distil-large-v3", "distil-medium.en", "distil-small.en"]

        add_field("Assistant Name:", "name_var", self.assistant_name)
        add_field("Wake Word:", "wake_var", self.wake_word)
        add_field("Backend Engine:", "engine_var", self.engine, options=["opv", "openclaw"])
        add_field("LLM Model:", "model_var", self.model_name, options=ollama_models)
        
        # Divider
        div = tk.Frame(form_frame, bg="#45475a", height=1)
        div.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(5, 15))
        row += 1

        add_field("STT Engine:", "stt_var", current_stt, options=["whisper", "vosk"])
        add_field("Whisper Model:", "whisper_model_var", current_whisper, options=whisper_models)
        
        # Divider
        div2 = tk.Frame(form_frame, bg="#45475a", height=1)
        div2.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(5, 15))
        row += 1

        add_field("TTS Engine:", "tts_var", current_tts, options=["piper", "pyttsx3", "edge", "system"])
        add_field("Piper Voice Path:", "piper_voice_var", current_piper)

        def _save():
            new_config = {
                "assistant_name": vars_dict["name_var"].get().strip() or "Assistant",
                "wake_word": vars_dict["wake_var"].get().strip() or "hey",
                "engine": vars_dict["engine_var"].get().strip(),
                "model": vars_dict["model_var"].get().strip() or self.model_name,
                "stt_engine": vars_dict["stt_var"].get().strip(),
                "whisper_model": vars_dict["whisper_model_var"].get().strip(),
                "tts": vars_dict["tts_var"].get().strip(),
                "piper_voice_path": vars_dict["piper_voice_var"].get().strip(),
            }
            if self.on_save_defaults:
                self.on_save_defaults(new_config)
            
            self.assistant_name = new_config["assistant_name"]
            self.wake_word = new_config["wake_word"]
            self.engine = new_config["engine"]
            self.model_name = new_config["model"]
            self.stt_engine = new_config["stt_engine"]
            self.whisper_model = new_config["whisper_model"]
            self.tts_engine = new_config["tts"]
            self.piper_voice = new_config["piper_voice_path"]

            if self.title_label:
                self.title_label.config(text=f"{self.assistant_name}")
            
            self._append_message("System", f"Saved configuration: Name='{self.assistant_name}', Model='{self.model_name}', Wake='{self.wake_word}'", "system_tag")
            dlg.destroy()

        btn_frame = tk.Frame(dlg, bg="#1e1e2e")
        btn_frame.pack(fill=tk.X, pady=(10, 20))
        
        save_btn = tk.Button(
            btn_frame,
            text="Save Changes",
            font=("Segoe UI", 11, "bold"),
            bg="#a6e3a1",
            fg="#11111b",
            activebackground="#b4befe",
            activeforeground="#11111b",
            relief=tk.FLAT,
            cursor="hand2",
            padx=16,
            pady=6,
            command=_save
        )
        save_btn.pack()

    def _open_memory_editor(self):
        """Open a popup to manage assistant memories."""
        try:
            from memory import load_memory, save_memory, add_memory_fact
        except ImportError:
            warn("memory module not found.")
            return

        win = tk.Toplevel(self.root)
        win.title("Memory Editor")
        win.geometry("520x500")
        win.configure(bg="#1e1e2e")
        win.attributes("-topmost", True)
        
        tk.Label(win, text="Manage Assistant Memory", font=("Segoe UI", 12, "bold"), bg="#1e1e2e", fg="#cdd6f4").pack(pady=10)

        main_frame = tk.Frame(win, bg="#1e1e2e")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        canvas = tk.Canvas(main_frame, bg="#1e1e2e", highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#1e1e2e")

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw", width=470)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        def refresh_memory_list():
            for widget in scrollable_frame.winfo_children():
                widget.destroy()
            
            facts = load_memory()
            if not facts:
                tk.Label(scrollable_frame, text="No memories found.", bg="#1e1e2e", fg="#a6adc8").pack(pady=20)
            
            for i, fact in enumerate(facts):
                row = tk.Frame(scrollable_frame, bg="#313244")
                row.pack(fill="x", pady=2, padx=5)
                
                lbl = tk.Label(row, text=fact, bg="#313244", fg="#cdd6f4", wraplength=380, justify="left", anchor="w")
                lbl.pack(side="left", fill="x", expand=True, padx=10, pady=5)
                
                def make_delete_cmd(idx=i):
                    def _del():
                        curr_facts = load_memory()
                        if idx < len(curr_facts):
                            curr_facts.pop(idx)
                            save_memory(curr_facts)
                            refresh_memory_list()
                    return _del
                
                del_btn = tk.Button(row, text="🗑", font=("Segoe UI", 10), bg="#f38ba8", fg="#11111b",
                                    relief=tk.FLAT, cursor="hand2", command=make_delete_cmd(i))
                del_btn.pack(side="right", padx=10, pady=5)

        refresh_memory_list()

        add_frame = tk.Frame(win, bg="#1e1e2e")
        add_frame.pack(fill="x", padx=10, pady=15)
        
        new_entry = tk.Entry(add_frame, font=("Segoe UI", 10), bg="#313244", fg="#cdd6f4", insertbackground="#cdd6f4")
        new_entry.pack(side="left", fill="x", expand=True, padx=(0, 10), ipady=4)
        
        def on_add():
            val = new_entry.get().strip()
            if val:
                add_memory_fact(val)
                new_entry.delete(0, tk.END)
                refresh_memory_list()
                
        add_btn = tk.Button(add_frame, text="➕ Add", font=("Segoe UI", 9, "bold"), bg="#a6e3a1", fg="#11111b",
                            relief=tk.FLAT, cursor="hand2", command=on_add)
        add_btn.pack(side="right", ipadx=5, ipady=2)

    def _on_close(self):
        self._is_closed = True
        if self.root:
            self.root.destroy()

    def _on_send_click(self, event=None):
        text = self.entry_var.get().strip()
        if text:
            self.entry_var.set("")
            self.post_user_message(text)
            if self.on_send_message:
                threading.Thread(target=self.on_send_message, args=(text,), daemon=True).start()

    def _append_message(self, sender: str, text: str, tag: str):
        if not self.chat_display or self._is_closed:
            return
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, f"\n{sender}: ", tag)
        self.chat_display.insert(tk.END, f"{text}\n", "msg_text")
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)

    def _process_queue(self):
        """Check queue for messages from background threads."""
        while not self.msg_queue.empty():
            try:
                task, args = self.msg_queue.get_nowait()
                if task == "user":
                    self._append_message("You", args[0], "user_tag")
                elif task == "ai":
                    self._append_message(self.assistant_name, args[0], "ai_tag")
                elif task == "system":
                    self._append_message("System", args[0], "system_tag")
                elif task == "think":
                    self._append_message("Thinking Process", args[0], "think_tag")
                elif task == "status":
                    if self.status_label:
                        self.status_label.config(text=args[0], fg=args[1])
            except Exception:
                pass

        if not self._is_closed and self.root:
            self.root.after(100, self._process_queue)

    def post_user_message(self, text: str):
        self.msg_queue.put(("user", (text,)))

    def post_ai_message(self, text: str):
        self.msg_queue.put(("ai", (text,)))

    def post_system_message(self, text: str):
        self.msg_queue.put(("system", (text,)))

    def post_think_message(self, text: str):
        self.msg_queue.put(("think", (text,)))

    def update_status(self, text: str, color: str = "#a6e3a1"):
        # Strip any remaining emojis from status strings before rendering
        clean_status = text.replace("⚡ ", "").replace("● ", "")
        self.msg_queue.put(("status", (clean_status, color)))
