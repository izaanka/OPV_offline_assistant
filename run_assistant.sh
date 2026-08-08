#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  🎙️  Local AI Voice Assistant — Launcher
#  Double-click this file in your file manager to start.
# ─────────────────────────────────────────────────────────────

# Resolve the directory this script lives in (works even with symlinks)
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"

# ── If we're NOT inside a terminal yet, re-launch inside one ──
if [ ! -t 1 ]; then
    # Try common terminal emulators in order of preference
    for TERM_EMU in gnome-terminal xterm konsole xfce4-terminal lxterminal mate-terminal tilix; do
        if command -v "$TERM_EMU" &>/dev/null; then
            case "$TERM_EMU" in
                gnome-terminal)
                    exec gnome-terminal -- bash -c "\"$0\"; exec bash"
                    ;;
                xterm)
                    exec xterm -hold -e bash -c "\"$0\""
                    ;;
                konsole)
                    exec konsole --hold -e bash -c "\"$0\""
                    ;;
                *)
                    exec "$TERM_EMU" -e bash -c "\"$0\"; exec bash"
                    ;;
            esac
        fi
    done
    # Last resort fallback
    xdg-open "$0"
    exit 0
fi

# ─────────────────────────────────────────────────────────────
#  From here on we are inside a terminal
# ─────────────────────────────────────────────────────────────

# Pretty colours
BOLD="\033[1m"; CYAN="\033[96m"; GREEN="\033[92m"
YELLOW="\033[93m"; RED="\033[91m"; RESET="\033[0m"

echo -e "${BOLD}${CYAN}"
echo "╔══════════════════════════════════════════════╗"
echo "║       🎙️  Local AI Voice Assistant           ║"
echo "╚══════════════════════════════════════════════╝"
echo -e "${RESET}"

cd "$SCRIPT_DIR" || { echo -e "${RED}Could not cd to $SCRIPT_DIR${RESET}"; read -rp "Press Enter to close..."; exit 1; }

# ── 1. Check & Auto-install System Packages (Linux/Debian/Ubuntu) ──────────────
if command -v apt-get &>/dev/null; then
    MISSING_PKGS=()
    if ! ldconfig -p 2>/dev/null | grep -i portaudio &>/dev/null && [ ! -f /usr/include/portaudio.h ]; then
        MISSING_PKGS+=("portaudio19-dev" "libportaudio2")
    fi
    if ! command -v espeak &>/dev/null; then
        MISSING_PKGS+=("espeak")
    fi
    if ! dpkg-query -W -f='${Status}' python3-venv 2>/dev/null | grep -q "ok installed"; then
        MISSING_PKGS+=("python3-venv" "python3-pip")
    fi
    if [ ! -f /usr/include/SDL2/SDL.h ] && ! command -v sdl2-config &>/dev/null; then
        MISSING_PKGS+=("libsdl2-dev")
    fi

    if [ ${#MISSING_PKGS[@]} -gt 0 ]; then
        echo -e "${YELLOW}[!] Missing system packages: ${MISSING_PKGS[*]}${RESET}"
        echo -e "${CYAN}[•] Installing missing system packages via sudo...${RESET}"
        sudo apt-get update && sudo apt-get install -y "${MISSING_PKGS[@]}" || \
        echo -e "${YELLOW}[!] System package installation failed or was skipped.${RESET}"
    else
        echo -e "${GREEN}[✓] System packages ready (portaudio, espeak, venv, sdl2).${RESET}"
    fi
fi

# ── 2. Check Python ─────────────────────────────────────────────────────────────
SYSTEM_PY=""
for PY in python3 python; do
    if command -v "$PY" &>/dev/null; then
        SYSTEM_PY="$PY"
        break
    fi
done

if [ -z "$SYSTEM_PY" ]; then
    echo -e "${RED}[✗] Python 3 not found. Please install python3.${RESET}"
    read -rp "Press Enter to close..."
    exit 1
fi
echo -e "${GREEN}[✓] System Python: $($SYSTEM_PY --version)${RESET}"

# ── 3. Setup & Activate Virtual Environment ─────────────────────────────────────
VENV_DIR="$SCRIPT_DIR/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${CYAN}[•] Creating virtual environment ($VENV_DIR)...${RESET}"
    $SYSTEM_PY -m venv "$VENV_DIR" || {
        echo -e "${YELLOW}[!] venv creation failed — retrying with sudo apt install python3-venv...${RESET}"
        sudo apt-get install -y python3-venv python3-pip
        $SYSTEM_PY -m venv "$VENV_DIR"
    }
fi

PYTHON="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"

# Ensure pip is up-to-date inside venv
$PIP install --upgrade pip setuptools wheel 2>/dev/null

# ── 4. Install Python Dependencies ──────────────────────────────────────────────
echo -e "${CYAN}[•] Installing/checking Python dependencies in virtualenv...${RESET}"
$PIP install -r "$SCRIPT_DIR/requirements.txt" || \
    $PIP install SpeechRecognition vosk pyttsx3 ollama sounddevice numpy piper-tts

# Attempt optional hardware/audio acceleration packages
$PIP install pygame 2>/dev/null || $PIP install pygame-ce 2>/dev/null || true
$PIP install pyaudio 2>/dev/null || true
echo -e "${GREEN}[✓] Python dependencies ready.${RESET}"

# ── 6. Download Piper Voice Models ──────────────────────────────────────────────
PIPER_DIR="$SCRIPT_DIR/piper-voices"
mkdir -p "$PIPER_DIR"

_dl_piper() {
    local NAME="$1" BASE="$2"
    local ONNX="$PIPER_DIR/${NAME}.onnx"
    if [ ! -f "$ONNX" ]; then
        echo -e "${CYAN}[•] Downloading Piper voice: ${NAME}...${RESET}"
        local OK=false
        if command -v curl &>/dev/null; then
            curl -fL --progress-bar "$BASE/${NAME}.onnx" -o "$ONNX" && \
            curl -fsSL "$BASE/${NAME}.onnx.json" -o "$ONNX.json" && OK=true
        elif command -v wget &>/dev/null; then
            wget -q --show-progress "$BASE/${NAME}.onnx" -O "$ONNX" && \
            wget -q "$BASE/${NAME}.onnx.json" -O "$ONNX.json" && OK=true
        fi
        $OK && echo -e "${GREEN}[✓] ${NAME} ready.${RESET}" \
             || echo -e "${YELLOW}[!] Download failed for ${NAME} — will fall back to espeak/pyttsx3.${RESET}"
    else
        echo -e "${GREEN}[✓] ${NAME} already present.${RESET}"
    fi
}

HF="https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0"
_dl_piper "en_US-ryan-high"   "$HF/en/en_US/ryan/high"   # ♂ male
_dl_piper "en_US-amy-medium"  "$HF/en/en_US/amy/medium"  # ♀ female

# ── 7. Start Ollama Server & Pull Default Model if needed ───────────────────────
if command -v ollama &>/dev/null; then
    if ! curl -sf http://localhost:11434 &>/dev/null; then
        echo -e "${CYAN}[•] Starting Ollama server in the background...${RESET}"
        ollama serve &>/tmp/ollama_serve.log &
        for i in {1..10}; do
            sleep 1
            if curl -sf http://localhost:11434 &>/dev/null; then
                echo -e "${GREEN}[✓] Ollama server ready.${RESET}"
                break
            fi
        done
    else
        echo -e "${GREEN}[✓] Ollama server running.${RESET}"
    fi

    # Check if any model exists, pull llama3.2:3b if none
    MODELS=$(ollama list 2>/dev/null | tail -n +2)
    if [ -z "$MODELS" ]; then
        echo -e "${CYAN}[•] No Ollama models found. Pulling default model llama3.2:3b...${RESET}"
        ollama pull llama3.2:3b
    fi
else
    echo -e "${YELLOW}[!] Ollama CLI not found. Download from https://ollama.com${RESET}"
fi

echo ""

# ── 8. Launch the Assistant ─────────────────────────────────────────────────────
echo -e "${GREEN}[🚀] Launching Voice Assistant...${RESET}\n"
$PYTHON "$SCRIPT_DIR/assistant.py" "$@"
EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -ne 0 ]; then
    echo -e "${RED}[✗] Assistant exited with code $EXIT_CODE.${RESET}"
fi

read -rp "Press Enter to close this window..."

