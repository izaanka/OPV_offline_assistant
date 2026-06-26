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

# ── Check Python ──────────────────────────────────────────────
PYTHON=""
for PY in python3 python; do
    if command -v "$PY" &>/dev/null; then
        PYTHON="$PY"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo -e "${RED}[✗] Python 3 not found. Please install it first.${RESET}"
    read -rp "Press Enter to close..."
    exit 1
fi
echo -e "${GREEN}[✓] Python: $($PYTHON --version)${RESET}"

# ── Ensure pip is available, bootstrap if missing ─────────────
echo -e "${CYAN}[•] Checking pip...${RESET}"

# Make sure ~/.local/bin (user pip installs) is on PATH
export PATH="$HOME/.local/bin:$PATH"

if ! $PYTHON -m pip --version &>/dev/null; then
    echo -e "${YELLOW}[!] pip not found — bootstrapping via ensurepip...${RESET}"
    if ! $PYTHON -m ensurepip --upgrade 2>/dev/null; then
        echo -e "${YELLOW}[!] ensurepip failed — downloading get-pip.py...${RESET}"
        GET_PIP=$(mktemp /tmp/get-pip-XXXXXX.py)
        if curl -fsSL https://bootstrap.pypa.io/get-pip.py -o "$GET_PIP" 2>/dev/null \
           || wget -qO "$GET_PIP" https://bootstrap.pypa.io/get-pip.py 2>/dev/null; then
            $PYTHON "$GET_PIP" --user
            rm -f "$GET_PIP"
        else
            echo -e "${RED}[✗] Could not install pip. Run:  sudo apt install python3-pip${RESET}"
            read -rp "Press Enter to close..."
            exit 1
        fi
    fi
fi
echo -e "${GREEN}[✓] pip: $($PYTHON -m pip --version 2>&1 | head -1)${RESET}"

# Detect if this Python uses PEP 668 (externally-managed env, e.g. Python 3.14 on Debian/Ubuntu)
# In that case we must pass --break-system-packages
PIP_FLAGS="--user"
if $PYTHON -m pip install --dry-run pip &>/dev/null; then
    : # normal pip
elif $PYTHON -m pip install --user --break-system-packages --dry-run pip &>/dev/null 2>&1; then
    PIP_FLAGS="--user --break-system-packages"
fi

# ── Install Python dependencies ───────────────────────────────
echo -e "${CYAN}[•] Installing/checking Python dependencies...${RESET}"
if ! $PYTHON -m pip install $PIP_FLAGS -r "$SCRIPT_DIR/requirements.txt"; then
    # Retry with --break-system-packages in case env is externally managed
    echo -e "${YELLOW}[!] Retrying with --break-system-packages...${RESET}"
    if ! $PYTHON -m pip install --user --break-system-packages -r "$SCRIPT_DIR/requirements.txt"; then
        echo -e "${RED}[✗] Failed to install dependencies from requirements.txt${RESET}"
        read -rp "Press Enter to close..."
        exit 1
    fi
fi
echo -e "${GREEN}[✓] Dependencies ready.${RESET}"

# ── Install Piper TTS + pygame (neural voice, fully offline) ───────────
echo -e "${CYAN}[•] Installing Piper TTS and pygame...${RESET}"
$PYTHON -m pip install $PIP_FLAGS piper-tts pygame 2>/dev/null \
    || $PYTHON -m pip install --user --break-system-packages piper-tts pygame 2>/dev/null \
    || echo -e "${YELLOW}[!] piper-tts install failed — TTS will fall back to espeak.${RESET}"

# ── Download Piper voice models (male + female) if not already present ────────
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
_dl_piper "en_US-amy-high"    "$HF/en/en_US/amy/high"    # ♀ female

# ── Start Ollama if not already running ───────────────────────
if ! curl -sf http://localhost:11434 &>/dev/null; then
    echo -e "${CYAN}[•] Starting Ollama server in the background...${RESET}"
    ollama serve &>/tmp/ollama_serve.log &
    OLLAMA_PID=$!
    # Give it a moment to boot
    for i in {1..10}; do
        sleep 1
        if curl -sf http://localhost:11434 &>/dev/null; then
            echo -e "${GREEN}[✓] Ollama server ready.${RESET}"
            break
        fi
        if [ "$i" -eq 10 ]; then
            echo -e "${YELLOW}[!] Ollama server may not be running. Check: ollama serve${RESET}"
        fi
    done
else
    echo -e "${GREEN}[✓] Ollama server already running.${RESET}"
fi

echo ""

# ── Build argument list ────────────────────────────────────────────────
CMD_ARGS=()

# ── Run the assistant ────────────────────────────────────────────────
$PYTHON "$SCRIPT_DIR/assistant.py" "${CMD_ARGS[@]}" "$@"
EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -ne 0 ]; then
    echo -e "${RED}[✗] Assistant exited with error code $EXIT_CODE.${RESET}"
fi

read -rp "Press Enter to close this window..."

