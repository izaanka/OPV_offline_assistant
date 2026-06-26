@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul 2>&1

:: ─────────────────────────────────────────────────────────────
::  🎙️  Local AI Voice Assistant — Windows Launcher
::  Double-click this file to start the assistant.
:: ─────────────────────────────────────────────────────────────

title Local AI Voice Assistant

echo.
echo  ╔══════════════════════════════════════════════╗
echo  ║       🎙️  Local AI Voice Assistant           ║
echo  ╚══════════════════════════════════════════════╝
echo.

:: Change to the directory this script lives in
cd /d "%~dp0"

:: ── Find Python ──────────────────────────────────────────────
set "PYTHON="
for %%P in (python python3) do (
    if "!PYTHON!"=="" (
        where %%P >nul 2>&1 && set "PYTHON=%%P"
    )
)

if "!PYTHON!"=="" (
    echo [X] Python not found.
    echo.
    echo     Download Python 3.10+ from: https://www.python.org/downloads/
    echo     Make sure to tick "Add Python to PATH" during install.
    echo.
    pause
    exit /b 1
)

for /f "tokens=*" %%V in ('!PYTHON! --version 2^>^&1') do echo [+] %%V

:: ── Ensure pip is available ───────────────────────────────────
echo [.] Checking pip...
!PYTHON! -m pip --version >nul 2>&1
if errorlevel 1 (
    echo [!] pip not found — bootstrapping...
    !PYTHON! -m ensurepip --upgrade >nul 2>&1
    if errorlevel 1 (
        echo [!] Downloading get-pip.py...
        curl -fsSL https://bootstrap.pypa.io/get-pip.py -o "%TEMP%\get-pip.py"
        if errorlevel 1 (
            echo [X] Could not download get-pip.py. Check your internet connection.
            pause
            exit /b 1
        )
        !PYTHON! "%TEMP%\get-pip.py"
        del "%TEMP%\get-pip.py" >nul 2>&1
    )
)
for /f "tokens=*" %%V in ('!PYTHON! -m pip --version 2^>^&1') do echo [+] pip: %%V

:: ── Install Python dependencies (excl. pyaudio — handled separately) ────────
echo [.] Installing core dependencies...
!PYTHON! -m pip install --user SpeechRecognition vosk pyttsx3 ollama
if errorlevel 1 (
    echo [X] Failed to install core dependencies.
    pause
    exit /b 1
)

:: ── Audio input: try PyAudio, fall back to sounddevice ────────
echo [.] Installing audio input library...
!PYTHON! -m pip install --user pyaudio >nul 2>&1
if not errorlevel 1 (
    echo [+] PyAudio installed successfully.
    goto audio_ready
)

echo [!] PyAudio failed (may need VC++ build tools).
echo [.] Trying pre-built wheel via pipwin...
!PYTHON! -m pip install --user pipwin >nul 2>&1
!PYTHON! -m pipwin install pyaudio >nul 2>&1
if not errorlevel 1 (
    echo [+] PyAudio installed via pipwin.
    goto audio_ready
)

echo [!] pipwin also failed. Trying sounddevice + numpy (no build needed)...
!PYTHON! -m pip install --user sounddevice numpy
if not errorlevel 1 (
    echo [+] sounddevice + numpy installed (PyAudio alternative).
    goto audio_ready
)

echo [X] Could not install any audio input library.
echo     Manual options:
echo       1. pip install pipwin ^&^& pipwin install pyaudio
echo       2. Download PyAudio wheel from https://www.lfd.uci.edu/~gohlke/pythonlibs/
echo       3. pip install sounddevice numpy
pause
exit /b 1

:audio_ready
echo [+] Audio input ready.
echo.

:: ── Install Piper TTS + pygame (neural voice, fully offline) ───────────
echo [.] Installing Piper TTS and pygame...
!PYTHON! -m pip install --user piper-tts pygame >nul 2>&1
if errorlevel 1 (
    echo [!] piper-tts install failed — TTS will fall back to SAPI/pyttsx3.
)

:: ── Download Piper voice models (male + female) if not already present ──────────
if not exist "%~dp0piper-voices\" mkdir "%~dp0piper-voices"

set "HF=https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0"

:: Male voice — Ryan (high)
if not exist "%~dp0piper-voices\en_US-ryan-high.onnx" (
    echo [.] Downloading Piper voice: en_US-ryan-high (~120 MB)...
    curl -fL "!HF!/en/en_US/ryan/high/en_US-ryan-high.onnx" -o "%~dp0piper-voices\en_US-ryan-high.onnx" >nul 2>&1
    curl -fsSL "!HF!/en/en_US/ryan/high/en_US-ryan-high.onnx.json" -o "%~dp0piper-voices\en_US-ryan-high.onnx.json" >nul 2>&1
    if exist "%~dp0piper-voices\en_US-ryan-high.onnx" ( echo [+] en_US-ryan-high ready. ) else ( echo [!] Ryan download failed - TTS will fall back. )
) else (
    echo [+] en_US-ryan-high already present.
)

:: Female voice — Amy (medium)
if not exist "%~dp0piper-voices\en_US-amy-high.onnx" (
    echo [.] Downloading Piper voice: en_US-amy-high (~120 MB)...
    curl -fL "!HF!/en/en_US/amy/high/en_US-amy-high.onnx" -o "%~dp0piper-voices\en_US-amy-high.onnx" >nul 2>&1
    curl -fsSL "!HF!/en/en_US/amy/high/en_US-amy-high.onnx.json" -o "%~dp0piper-voices\en_US-amy-high.onnx.json" >nul 2>&1
    if exist "%~dp0piper-voices\en_US-amy-high.onnx" ( echo [+] en_US-amy-high ready. ) else ( echo [!] Amy download failed - TTS will fall back. )
) else (
    echo [+] en_US-amy-high already present.
)
echo.

:: ── Check / start Ollama ─────────────────────────────────────
echo [.] Checking Ollama server...
curl -sf http://localhost:11434 >nul 2>&1
if errorlevel 1 (
    echo [.] Starting Ollama server in the background...
    where ollama >nul 2>&1
    if errorlevel 1 (
        echo [!] Ollama not found. Download from: https://ollama.com/download
        echo     Install it, then run this script again.
        pause
        exit /b 1
    )
    start /b "" ollama serve
    :: Wait for it to start
    set /a TRIES=0
    :wait_ollama
    timeout /t 1 /nobreak >nul
    curl -sf http://localhost:11434 >nul 2>&1
    if not errorlevel 1 goto ollama_ready
    set /a TRIES+=1
    if !TRIES! lss 10 goto wait_ollama
    echo [!] Ollama server may not have started — proceeding anyway.
    goto run_assistant
    :ollama_ready
    echo [+] Ollama server ready.
) else (
    echo [+] Ollama server already running.
)

:run_assistant
echo.

:: ── Build argument list ─────────────────────────────────────────────────────
set "CMD_ARGS="

:: ── Run the assistant ─────────────────────────────────────────
!PYTHON! "%~dp0assistant.py" !CMD_ARGS! %*
set EXIT_CODE=%errorlevel%

echo.
if not "%EXIT_CODE%"=="0" (
    echo [X] Assistant exited with error code %EXIT_CODE%.
)

pause
endlocal

