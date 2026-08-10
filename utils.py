import sys
from datetime import datetime

RESET   = "\033[0m"
BOLD    = "\033[1m"
CYAN    = "\033[96m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
MAGENTA = "\033[95m"
DIM     = "\033[2m"
GRAY    = "\033[90m"

def _get_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

def c(text: str, colour: str) -> str: 
    """Colorize text with ANSI escape codes."""
    return f"{colour}{text}{RESET}"

def _log(level: str, msg: str, color: str, file=sys.stdout) -> None:
    ts = c(f"[{_get_timestamp()}]", GRAY)
    lvl = c(f"[{level}]", color)
    print(f"{ts} {lvl} {msg}", file=file)

def info(msg: str) -> None:
    """Print an info message."""
    _log("INFO", msg, CYAN)
    
def success(msg: str) -> None:
    """Print a success message."""
    _log("SUCCESS", msg, GREEN)
    
def warn(msg: str) -> None:
    """Print a warning message."""
    _log("WARN", msg, YELLOW)
    
def error(msg: str) -> None:
    """Print an error message to stderr."""
    _log("ERROR", msg, RED, file=sys.stderr)
    
def speak_label(msg: str) -> None:
    """Print a speaking label."""
    _log("SPEAK", msg, MAGENTA)
    
def user_label(msg: str) -> None:
    """Print a user label."""
    _log("USER", msg, GREEN)
    
def ai_label(msg: str) -> None:
    """Print an AI label."""
    _log("AI", msg, CYAN)
