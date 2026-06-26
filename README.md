# 🎙️ Local AI Voice Assistant

A fully local, privacy-respecting voice assistant that runs on your machine using **Ollama** for the LLM brain.

---

## How It Works

```
Microphone → Wake Word ("hey") → Speech-to-Text → Ollama LLM → Text-to-Speech → Speaker
```

| Stage | Tool |
|---|---|
| Wake Word | Keyword match via SpeechRecognition |
| STT | Google Speech API (online) or Sphinx (offline fallback) |
| LLM | Ollama (any local model) |
| TTS | pyttsx3 (offline) or edge-tts (high quality) |

---

## Prerequisites

### 1. System packages (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install portaudio19-dev espeak python3-dev python3-pip
```

### 2. Ollama
Install from https://ollama.com and pull a model:
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3          # or: mistral, phi3, gemma2, etc.
ollama serve                # start the server (runs in background)
```

### 3. Python dependencies
```bash
cd "local ai chatbot"
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

```bash
# Basic — uses interactive model picker, "hey" wake word, and Piper neural TTS
python assistant.py

# Use a specific model (skips the picker)
python assistant.py --model mistral

# Change the wake word
python assistant.py --wake-word jarvis

# Use Edge TTS instead of Piper (requires internet)
python assistant.py --tts edge --edge-voice en-US-JennyNeural

# Slower speech rate (for pyttsx3 fallback)
python assistant.py --rate 150

# List available TTS voices
python assistant.py --list-voices
```

---

## Voice Commands

| What you say | What happens |
|---|---|
| **"hey"** | Wakes the assistant |
| **"hey [your question]"** | Wakes + asks inline |
| **"reset"** / **"clear history"** | Clears conversation memory |
| **"quit"** / **"bye"** | Exits the assistant |

---

## Troubleshooting

**Microphone not working?**
```bash
arecord -l        # list audio devices
python3 -c "import speech_recognition as sr; print(sr.Microphone.list_microphone_names())"
```

**PyAudio install fails?**
```bash
sudo apt install portaudio19-dev
pip install pyaudio
```

**pyttsx3 not speaking on Linux?**
```bash
sudo apt install espeak espeak-data libespeak1
```

**Ollama model not found?**
```bash
ollama list                   # see installed models
ollama pull llama3            # download llama3
```

---

**Piper TTS not working?**
Ensure `piper-tts` and `pygame` are installed, and the `.onnx` voice model exists in `piper-voices/`.
The launchers auto-download the `en_US-ryan-high` model (~120MB) for you.

---

## High-Quality Local TTS (Piper)

By default, the assistant now uses **Piper TTS** for extremely natural, human-like voice synthesis that runs 100% offline. The launchers handle downloading the required model automatically.

If Piper fails or is unavailable, it gracefully falls back to:
1. `pyttsx3`
2. Native system TTS (`say`, SAPI, `espeak`)
3. `edge-tts` (if configured and online)
