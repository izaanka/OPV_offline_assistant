#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  🎙️  Local AI Voice Assistant — macOS Launcher
#
#  To use: right-click → Open (first time only, to bypass Gatekeeper)
#  After that: just double-click from Finder.
# ─────────────────────────────────────────────────────────────

# Resolve directory this script lives in
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0" 2>/dev/null || echo "$0")")" && pwd)"

# ── If not in a terminal, open one ───────────────────────────
if [ ! -t 1 ]; then
    osascript -e "tell application \"Terminal\" to do script \"bash '${SCRIPT_DIR}/run_assistant_mac.command'\""
    exit 0
fi

# Pretty colours
BOLD="\033[1m"; CYAN="\033[96m"; GREEN="\033[92m"
YELLOW="\033[93m"; RED="\033[91m"; RESET="\033[0m"

echo -e "${BOLD}${CYAN}"
echo "╔══════════════════════════════════════════════╗"
echo "║       🎙️  Local AI Voice Assistant           ║"
echo "╚══════════════════════════════════════════════╝"
echo -e "${RESET}"

cd "$SCRIPT_DIR" || { echo -e "${RED}Could not cd to $SCRIPT_DIR${RESET}"; read -rp "Press Enter..."; exit 1; }

# ── Find Python ───────────────────────────────────────────────
PYTHON=""
for PY in python3 python; do
    if command -v "$PY" &>/dev/null; then
        PYTHON="$PY"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo -e "${RED}[✗] Python 3 not found.${RESET}"
    echo ""
    echo "  Install options:"
    echo "    • Homebrew:  brew install python"
    echo "    • Official:  https://www.python.org/downloads/macos/"
    echo ""
    # Open the download page automatically
    open "https://www.python.org/downloads/macos/" 2>/dev/null
    read -rp "Press Enter to close..."
    exit 1
fi
echo -e "${GREEN}[✓] $($PYTHON --version)${RESET}"

# ── Add user local bin to PATH ────────────────────────────────
export PATH="$HOME/Library/Python/$(${PYTHON} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')/bin:$HOME/.local/bin:/usr/local/bin:$PATH"

# ── Homebrew: install portaudio (needed for PyAudio) ─────────
echo -e "${CYAN}[•] Checking audio dependencies (portaudio)...${RESET}"
if command -v brew &>/dev/null; then
    if ! brew list portaudio &>/dev/null; then
        echo -e "${YELLOW}[!] Installing portaudio via Homebrew (required for microphone)...${RESET}"
        brew install portaudio
    else
        echo -e "${GREEN}[✓] portaudio already installed.${RESET}"
    fi
else
    echo -e "${YELLOW}[!] Homebrew not found. If PyAudio fails, install Homebrew first:${RESET}"
    echo -e "${YELLOW}    /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"${RESET}"
    echo -e "${YELLOW}    Then run: brew install portaudio${RESET}"
fi

# ── Ensure pip is available ───────────────────────────────────
echo -e "${CYAN}[•] Checking pip...${RESET}"
if ! $PYTHON -m pip --version &>/dev/null; then
    echo -e "${YELLOW}[!] pip not found — bootstrapping...${RESET}"
    if ! $PYTHON -m ensurepip --upgrade 2>/dev/null; then
        echo -e "${YELLOW}[!] Downloading get-pip.py...${RESET}"
        GET_PIP=$(mktemp /tmp/get-pip-XXXXXX.py)
        if curl -fsSL https://bootstrap.pypa.io/get-pip.py -o "$GET_PIP"; then
            $PYTHON "$GET_PIP" --user
            rm -f "$GET_PIP"
        else
            echo -e "${RED}[✗] Could not install pip.${RESET}"
            echo "    Try: sudo easy_install pip  OR  brew install python"
            read -rp "Press Enter to close..."
            exit 1
        fi
    fi
fi
echo -e "${GREEN}[✓] pip: $($PYTHON -m pip --version 2>&1 | head -1)${RESET}"

# ── Install core Python dependencies ─────────────────────────
echo -e "${CYAN}[•] Installing core Python dependencies...${RESET}"
if ! $PYTHON -m pip install --user SpeechRecognition vosk pyttsx3 ollama; then
    echo -e "${RED}[✗] Failed to install core dependencies.${RESET}"
    read -rp "Press Enter to close..."
    exit 1
fi

# ── Audio input: try PyAudio, fall back to sounddevice ────────
echo -e "${CYAN}[•] Installing audio input library...${RESET}"
if $PYTHON -m pip install --user pyaudio &>/dev/null; then
    echo -e "${GREEN}[✓] PyAudio installed.${RESET}"
else
    echo -e "${YELLOW}[!] PyAudio failed. Trying sounddevice + numpy (no portaudio needed)...${RESET}"
    if $PYTHON -m pip install --user sounddevice numpy; then
        echo -e "${GREEN}[✓] sounddevice + numpy installed (PyAudio alternative).${RESET}"
    else
        echo -e "${RED}[✗] Could not install any audio input library.${RESET}"
        echo "  Options:"
        echo "    brew install portaudio && pip install pyaudio"
        echo "  or:"
        echo "    pip install sounddevice numpy"
        read -rp "Press Enter to close..."
        exit 1
    fi
fi
echo -e "${GREEN}[✓] Dependencies ready.${RESET}"

# ── Install Piper TTS + pygame (neural voice, fully offline) ───────────
echo -e "${CYAN}[•] Installing Piper TTS and pygame...${RESET}"
if ! $PYTHON -m pip install --user piper-tts pygame 2>/dev/null; then
    echo -e "${YELLOW}[!] piper-tts install failed — TTS will fall back to say/espeak.${RESET}"
fi

# ── Download Piper voice models (male + female) if not already present ────────
PIPER_DIR="$SCRIPT_DIR/piper-voices"
mkdir -p "$PIPER_DIR"

_dl_piper() {
    local NAME="$1" BASE="$2"
    local ONNX="$PIPER_DIR/${NAME}.onnx"
    if [ ! -f "$ONNX" ]; then
        echo -e "${CYAN}[•] Downloading Piper voice: ${NAME}...${RESET}"
        if curl -fL --progress-bar "$BASE/${NAME}.onnx" -o "$ONNX" && \
           curl -fsSL "$BASE/${NAME}.onnx.json" -o "$ONNX.json"; then
            echo -e "${GREEN}[✓] ${NAME} ready.${RESET}"
        else
            echo -e "${YELLOW}[!] Download failed for ${NAME} — will fall back to system TTS.${RESET}"
        fi
    else
        echo -e "${GREEN}[✓] ${NAME} already present.${RESET}"
    fi
}

HF="https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0"
_dl_piper "en_US-ryan-high"   "$HF/en/en_US/ryan/high"   # ♂ male
_dl_piper "en_US-amy-medium"  "$HF/en/en_US/amy/medium"  # ♀ female

# ── Check / start Ollama ──────────────────────────────────────
echo -e "${CYAN}[•] Checking Ollama server...${RESET}"
if ! curl -sf http://localhost:11434 &>/dev/null; then
    if command -v ollama &>/dev/null; then
        echo -e "${CYAN}[•] Starting Ollama server in the background...${RESET}"
        ollama serve &>/tmp/ollama_serve.log &
        for i in {1..10}; do
            sleep 1
            if curl -sf http://localhost:11434 &>/dev/null; then
                echo -e "${GREEN}[✓] Ollama server ready.${RESET}"
                break
            fi
            if [ "$i" -eq 10 ]; then
                echo -e "${YELLOW}[!] Ollama server may not have started. Check: ollama serve${RESET}"
            fi
        done
    else
        echo -e "${RED}[✗] Ollama not found.${RESET}"
        echo ""
        echo "  Install Ollama for macOS: https://ollama.com/download/mac"
        open "https://ollama.com/download/mac" 2>/dev/null
        read -rp "Press Enter to close..."
        exit 1
    fi
else
    echo -e "${GREEN}[✓] Ollama server already running.${RESET}"
fi

echo ""

# ── Build argument list (array handles spaces in paths) ────────────────────
CMD_ARGS=()

# ── Run the assistant ─────────────────────────────────────────────────
$PYTHON "$SCRIPT_DIR/assistant.py" "${CMD_ARGS[@]}" "$@"
EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -ne 0 ]; then
    echo -e "${RED}[✗] Assistant exited with error code $EXIT_CODE.${RESET}"
fi

read -rp "Press Enter to close this window..."
