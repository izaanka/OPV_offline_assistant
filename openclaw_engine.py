import os
import re
import urllib.request
import json as _json
from typing import List, Dict, Optional, Callable

from utils import info, success, warn, error
from memory import load_memory

try:
    import openclaw_sdk
    from openclaw_sdk import OpenClawClient, AgentConfig, Agent
    OPENCLAW_AVAILABLE = True
except ImportError:
    OPENCLAW_AVAILABLE = False


class OpenClawLLM:
    """Wrapper for the OpenClaw agentic framework engine."""

    def __init__(self, model: str = "llama3", assistant_name: str = "Assistant"):
        self.model = model
        self.assistant_name = assistant_name
        self.history: List[Dict[str, str]] = []
        
        # Initialize OpenClaw SDK if available
        if OPENCLAW_AVAILABLE:
            try:
                self.client = OpenClawClient()
                self.agent = Agent(
                    config=AgentConfig(
                        name=self.assistant_name,
                        model=self.model,
                        backend="ollama",  # Connect to local ollama instance
                        capabilities=["computer_use", "web_browser", "file_system"]
                    )
                )
            except Exception as e:
                warn(f"Failed to initialize OpenClaw agent: {e}")
                self.agent = None
        else:
            self.agent = None

        self.reset()

    def reset(self) -> None:
        """Reset conversation history."""
        self.history = []

    def check_available(self) -> bool:
        """Check if OpenClaw SDK and backend are available."""
        if not OPENCLAW_AVAILABLE:
            return False
        
        # We also need Ollama to be running if it's the backend
        try:
            req = urllib.request.Request("http://localhost:11434/api/tags")
            with urllib.request.urlopen(req, timeout=2) as resp:
                return resp.status == 200
        except Exception:
            return False

    def warmup(self) -> None:
        """Warmup OpenClaw engine."""
        info("Warming up OpenClaw engine...")
        if self.agent and hasattr(self.agent, "warmup"):
            try:
                self.agent.warmup()
            except Exception as e:
                warn(f"OpenClaw warmup failed: {e}")
        success("OpenClaw engine ready.")

    def chat(self, user_input: str, on_think: Optional[Callable[[str], None]] = None) -> str:
        """Process user input using the OpenClaw autonomous loop."""
        self.history.append({"role": "user", "content": user_input})
        
        if not OPENCLAW_AVAILABLE or not self.agent:
            return "Error: OpenClaw SDK is not available or failed to initialize."
            
        try:
            # Emulate an event stream loop if on_think is provided
            response_text = ""
            if hasattr(self.agent, "stream"):
                for event in self.agent.stream(user_input):
                    if type(event).__name__ == "ThinkingEvent" and on_think:
                        on_think(event.text)
                    elif type(event).__name__ == "ContentEvent":
                        response_text += event.text
            else:
                # Fallback to synchronous execution
                result = self.agent.execute(user_input)
                response_text = getattr(result, "content", str(result))
                
            self.history.append({"role": "assistant", "content": response_text})
            return response_text
        except Exception as e:
            error(f"OpenClaw execution error: {e}")
            return f"I'm sorry, my OpenClaw agent encountered an error: {e}"
