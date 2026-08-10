"""Terminal command execution module for OPV Voice Assistant — Full shell command execution."""

import subprocess
from typing import Dict, Any

from modules_registry import BaseModule
from utils import info, warn


class TerminalModule(BaseModule):
    name = "terminal"
    description = (
        "Run shell commands on the local machine (Linux/macOS/Windows). "
        "Parameters: {\"command\": \"shell_command_line\"}. ALWAYS requires user pop-up confirmation."
    )
    requires_confirmation = True  # Always ask user confirmation via GUI pop-up before executing shell commands!

    def can_handle_direct(self, user_input: str) -> bool:
        lower = user_input.lower()
        triggers = ["run command", "execute command", "run in terminal", "terminal ", "shell ", "bash "]
        return any(lower.startswith(t) or f" {t}" in lower for t in triggers)

    def parse_direct_args(self, user_input: str) -> Dict[str, Any]:
        lower = user_input.lower()
        triggers = ["run command", "execute command", "run in terminal", "terminal", "shell", "bash"]
        cmd = user_input
        for t in triggers:
            if t in lower:
                cmd = user_input[lower.find(t) + len(t):].strip()
                break
        return {"command": cmd}

    def execute(self, params: Dict[str, Any], user_input: str = "") -> str:
        cmd = params.get("command", "").strip()
        if not cmd and user_input:
            cmd = self.parse_direct_args(user_input).get("command", "")

        if not cmd:
            return "Error: No command string was provided to execute."

        info(f"Executing shell command: {cmd}")
        try:
            res = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )

            stdout = res.stdout.strip() if res.stdout else ""
            stderr = res.stderr.strip() if res.stderr else ""

            output_lines = []
            if stdout:
                output_lines.append(f"[STDOUT]:\n{stdout}")
            if stderr:
                output_lines.append(f"[STDERR]:\n{stderr}")
            if res.returncode != 0:
                output_lines.append(f"[Exit code]: {res.returncode}")

            output = "\n".join(output_lines)
            if not output:
                output = f"Command '{cmd}' completed with exit code 0 (no output)."

            # Truncate output if excessively long
            if len(output) > 3000:
                output = output[:3000] + "... [output truncated]"

            return f"Terminal Execution Result for (`{cmd}`):\n{output}"

        except subprocess.TimeoutExpired:
            return f"Error: Command '{cmd}' timed out after 30 seconds."
        except Exception as e:
            warn(f"Terminal execution error: {e}")
            return f"Error executing terminal command '{cmd}': {e}"
