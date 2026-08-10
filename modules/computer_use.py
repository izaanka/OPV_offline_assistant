import os
import time
from typing import Dict, Any
import subprocess

from utils import info, success, error, warn
from modules import BaseModule, modules_registry

try:
    import pyautogui
    pyautogui.FAILSAFE = False
except ImportError:
    pyautogui = None

class ComputerUseModule(BaseModule):
    name = "computer_use"
    description = (
        "Agentic workflow tool for computer automation (mouse, keyboard, screen). "
        "Allows you to control the OS UI directly. "
        "Parameters: {\"action\": \"mouse_move|mouse_click|type_text|press_key|hotkey|screen_size\", "
        "\"x\": int, \"y\": int, \"text\": str, \"key\": str, \"keys\": [str]}"
    )
    requires_confirmation = False

    def can_handle_direct(self, user_input: str) -> bool:
        lower = user_input.lower().strip()
        triggers = ["move mouse", "click", "type text", "press key", "screen size"]
        return any(t in lower for t in triggers)

    def parse_direct_args(self, user_input: str) -> Dict[str, Any]:
        return {"action": "screen_size"}

    def execute(self, params: Dict[str, Any], user_input: str = "") -> str:
        if not pyautogui:
            return "Error: pyautogui is not installed. Computer automation is disabled."

        action = params.get("action", "").lower()

        try:
            if action == "screen_size":
                w, h = pyautogui.size()
                return f"Screen resolution: {w}x{h}"

            elif action == "mouse_move":
                x = int(params.get("x", 0))
                y = int(params.get("y", 0))
                pyautogui.moveTo(x, y, duration=0.25)
                return f"Mouse moved to ({x}, {y})."

            elif action == "mouse_click":
                x = params.get("x")
                y = params.get("y")
                button = params.get("button", "left")
                if x is not None and y is not None:
                    pyautogui.click(x=int(x), y=int(y), button=button)
                    return f"Clicked {button} at ({x}, {y})."
                else:
                    pyautogui.click(button=button)
                    return f"Clicked {button} at current mouse position."

            elif action == "type_text":
                text = params.get("text", "")
                if not text:
                    return "Error: 'text' parameter is required."
                pyautogui.write(text, interval=0.01)
                return f"Typed text: '{text}'"

            elif action == "press_key":
                key = params.get("key", "")
                if not key:
                    return "Error: 'key' parameter is required."
                pyautogui.press(key)
                return f"Pressed key '{key}'."

            elif action == "hotkey":
                keys = params.get("keys", [])
                if not keys or not isinstance(keys, list):
                    return "Error: 'keys' parameter must be a list of keys."
                pyautogui.hotkey(*keys)
                return f"Pressed hotkey: {'+'.join(keys)}"

            else:
                return f"Error: Unknown action '{action}'."
        except Exception as e:
            error(f"Computer Use error: {e}")
            return f"Computer Use module encountered an error: {e}"

# Register the module
modules_registry.register(ComputerUseModule)
