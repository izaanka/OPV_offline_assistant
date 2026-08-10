"""LLM module for OPV voice assistant — Ollama wrapper with memory-aware prompting."""

import re
import json as _json
import urllib.request
from typing import Optional, List, Dict

from utils import c, info, success, warn, error, BOLD, CYAN, GREEN, DIM, RESET
from config import CONVERSATION_HIST, DEFAULT_MODEL
from memory import load_memory
import modules_registry

try:
    import ollama as ollama_client
    OLLAMA_LIB_AVAILABLE = True
except ImportError:
    OLLAMA_LIB_AVAILABLE = False


class OllamaLLM:
    """Wrapper for local Ollama LLM with conversation history and memory-aware prompting."""

    def __init__(self, model: str = DEFAULT_MODEL, assistant_name: str = "Assistant"):
        self.model = model
        self.assistant_name = assistant_name
        self.history: List[Dict[str, str]] = []
        self.reset()

    def reset(self) -> None:
        """Reset conversation history to initial system prompt."""
        self.history = [{"role": "system", "content": self._get_system_prompt()}]

    def _get_system_prompt(self) -> str:
        """Build dynamic system prompt with assistant name, memory facts, and available tools."""
        facts = load_memory()  # returns list of strings
        facts_block = ""
        if facts:
            facts_block = "\n\nHere are things you know about the user:\n" + \
                          "\n".join(f"- {f}" for f in facts)

        tool_instructions = modules_registry.get_system_prompt_tool_instructions()

        self_tooling_instructions = """

--- SELF-TOOLING CAPABILITY ---
You have the ability to CREATE NEW TOOLS when you encounter a task that no existing tool can handle.
Use the `tool_creator` module with these actions:
1. write_tool: Write a new tool module. Provide tool_name, description, code (Python execute body), parameters (JSON schema dict), and optionally requires_confirmation.
2. test_tool: Test the tool you just created. Provide tool_name and test_params.
3. register_tool: Register the tested tool so it becomes available immediately. Provide tool_name.
4. read_file: Read any project file to inspect existing modules or error logs.
5. run_shell: Run a shell command (e.g., pip install a missing package).

Workflow when you need a capability that doesn't exist:
TOOL_CALL: tool_creator {"action": "write_tool", "tool_name": "...", "description": "...", "code": "...", "parameters": {...}}
→ Then: TOOL_CALL: tool_creator {"action": "test_tool", "tool_name": "...", "test_params": {...}}
→ If test fails, read the error, fix the code with another write_tool call
→ Then: TOOL_CALL: tool_creator {"action": "register_tool", "tool_name": "..."}
→ Finally: Use the newly registered tool with: TOOL_CALL: <tool_name> {params}

Only create new tools when no existing tool can accomplish the task. Always test before registering.
"""

        import os
        home_dir = os.path.expanduser("~")
        
        return (
            f"You are {self.assistant_name}, a helpful AI voice assistant running locally with live web and system tool capabilities. "
            f"You may include your step-by-step reasoning inside <think>...</think> tags prior to your response or tool call. "
            f"Keep your final answer VERY short, concise, and conversational — exactly as spoken by a human. "
            f"Never claim that you cannot view webpages or access real-time information, because live module context and web tools are provided to you. "
            f"You HAVE the ability to control the user's mouse and keyboard via the computer_use tool. Never claim that you cannot control the computer or mouse. "
            f"CRITICAL: Do NOT use asterisks (*), markdown formatting, bold text, italics, or emojis in your final spoken response under ANY circumstances."
            f"\n\n--- ENVIRONMENT ---\n"
            f"Your current running environment home directory is: {home_dir}\n"
            f"The operating system username is: {os.environ.get('USER', 'user')}\n"
            f"Do not guess or hallucinate user directories (like /home/Aizen). Only search within valid, real paths.\n"
            f"{facts_block}"
            f"{tool_instructions}"
            f"{self_tooling_instructions}"
        )

    def warmup(self) -> None:
        """Send a tiny prompt to pre-load the model into VRAM."""
        info(f"Warming up model {self.model} into VRAM...")
        try:
            if OLLAMA_LIB_AVAILABLE:
                ollama_client.chat(model=self.model,
                                   messages=[{"role": "user", "content": "hi"}],
                                   options={"num_predict": 1})
            else:
                data = _json.dumps({
                    "model": self.model,
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                    "options": {"num_predict": 1}
                }).encode()
                req = urllib.request.Request(
                    "http://localhost:11434/api/chat",
                    data=data, headers={"Content-Type": "application/json"}
                )
                urllib.request.urlopen(req, timeout=30)
            success("Model warmed up and ready.")
        except Exception as e:
            warn(f"Warmup failed: {e}")

    def _trim_history(self) -> None:
        """Keep history within configured limits."""
        max_len = 1 + (CONVERSATION_HIST * 2)
        if len(self.history) > max_len:
            self.history = [self.history[0]] + self.history[-(max_len - 1):]

    def chat(self, user_input: str, on_think: Optional[Callable[[str], None]] = None) -> str:
        """Process user input: fetch module context, generate response, execute tools, store in history."""
        # Get direct module context (e.g. direct triggers)
        module_ctx = modules_registry.get_direct_context(user_input)
        if module_ctx:
            if "[SILENT_SUCCESS]" in module_ctx:
                clean_response = module_ctx.replace("[file_manager context]:", "").strip()
                self.history.append({"role": "user", "content": user_input})
                self.history.append({"role": "assistant", "content": clean_response})
                return clean_response
            augmented = f"{user_input}\n\n[System context — live module data:\n{module_ctx}]"
        else:
            augmented = user_input

        self.history.append({"role": "user", "content": augmented})
        self._trim_history()

        # Refresh system prompt (memory/tools may have changed)
        self.history[0] = {"role": "system", "content": self._get_system_prompt()}

def _try_parse_json_dict(s: str) -> Optional[dict]:
    if not isinstance(s, str):
        return None
    s_clean = s.strip()
    if not s_clean:
        return None

    candidates = [s_clean, s_clean.replace(r'\"', '"')]
    for cand in candidates:
        try:
            res = _json.loads(cand, strict=False)
            if isinstance(res, dict):
                return res
        except Exception:
            pass

        try:
            decoder = _json.JSONDecoder(strict=False)
            res, _ = decoder.raw_decode(cand)
            if isinstance(res, dict):
                return res
        except Exception:
            pass

    return None


def _normalize_params_dict(params: dict) -> dict:
    if isinstance(params, dict):
        if "action" not in params and "arg" in params:
            arg_val = params["arg"]
            if isinstance(arg_val, str):
                parsed = _try_parse_json_dict(arg_val)
                if parsed and "action" in parsed:
                    return parsed
            elif isinstance(arg_val, dict) and "action" in arg_val:
                return arg_val
    return params


def extract_tool_call(text: str):
    """Robustly extract (module_name, params_dict) from response_text handling nested JSON objects, code strings, escaped quotes, and arg wrappers."""
    match = re.search(r'TOOL_CALL:\s*([a-zA-Z0-9_]+)', text)
    if not match:
        return None, None

    module_name = match.group(1).strip()
    after_mod = text[match.end():]

    params = {}
    brace_idx = after_mod.find('{')
    if brace_idx != -1:
        json_candidate = after_mod[brace_idx:].strip()
        parsed = _try_parse_json_dict(json_candidate)
        if parsed:
            params = parsed
        else:
            # Fallback: manual balanced brace extraction
            depth = 0
            in_string = False
            escape = False
            end_idx = -1
            for i, ch in enumerate(json_candidate):
                if escape:
                    escape = False
                    continue
                if ch == '\\':
                    escape = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if not in_string:
                    if ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            end_idx = i
                            break
            if end_idx != -1:
                raw_block = json_candidate[:end_idx+1]
                parsed_block = _try_parse_json_dict(raw_block)
                if parsed_block:
                    params = parsed_block

    if not params:
        line_match = re.search(r'TOOL_CALL:\s*([a-zA-Z0-9_]+)\s*(.*)', text, re.DOTALL)
        if line_match:
            raw = line_match.group(2).strip()
            parsed_raw = _try_parse_json_dict(raw)
            if parsed_raw:
                params = parsed_raw
            elif raw:
                params = {"arg": raw}

    params = _normalize_params_dict(params)
    return module_name, params


class OllamaLLM:
    """Wrapper for local Ollama LLM with conversation history and memory-aware prompting."""

    def __init__(self, model: str = DEFAULT_MODEL, assistant_name: str = "Assistant"):
        self.model = model
        self.assistant_name = assistant_name
        self.history: List[Dict[str, str]] = []
        self.reset()

    def reset(self) -> None:
        """Reset conversation history to initial system prompt."""
        self.history = [{"role": "system", "content": self._get_system_prompt()}]

    def _get_system_prompt(self) -> str:
        """Build dynamic system prompt with assistant name, memory facts, and available tools."""
        facts = load_memory()  # returns list of strings
        facts_block = ""
        if facts:
            facts_block = "\n\nHere are things you know about the user:\n" + \
                          "\n".join(f"- {f}" for f in facts)

        tool_instructions = modules_registry.get_system_prompt_tool_instructions()

        self_tooling_instructions = """

--- SELF-TOOLING CAPABILITY ---
You have the ability to CREATE NEW TOOLS when you encounter a task that no existing tool can handle.
Use the `tool_creator` module with these actions:
1. write_tool: Write a new tool module. Provide tool_name, description, code (Python execute body), parameters (JSON schema dict), and optionally requires_confirmation.
2. test_tool: Test the tool you just created. Provide tool_name and test_params.
3. register_tool: Register the tested tool so it becomes available immediately. Provide tool_name.
4. read_file: Read any project file to inspect existing modules or error logs.
5. run_shell: Run a shell command (e.g., pip install a missing package).

Workflow when you need a capability that doesn't exist:
TOOL_CALL: tool_creator {"action": "write_tool", "tool_name": "...", "description": "...", "code": "...", "parameters": {...}}
→ Then: TOOL_CALL: tool_creator {"action": "test_tool", "tool_name": "...", "test_params": {...}}
→ If test fails, read the error, fix the code with another write_tool call
→ Then: TOOL_CALL: tool_creator {"action": "register_tool", "tool_name": "..."}
→ Finally: Use the newly registered tool with: TOOL_CALL: <tool_name> {params}

Only create new tools when no existing tool can accomplish the task. Always test before registering.
"""

        return (
            f"You are {self.assistant_name}, a helpful AI voice assistant running locally with live web and system tool capabilities. "
            f"You may include your step-by-step reasoning inside <think>...</think> tags prior to your response or tool call. "
            f"Keep your final answer VERY short, concise, and conversational — exactly as spoken by a human. "
            f"Never claim that you cannot view webpages or access real-time information, because live module context and web tools are provided to you. "
            f"When live system context is provided, use it directly to answer the user's query. "
            f"You HAVE the ability to control the user's mouse and keyboard via the computer_use tool. Never claim that you cannot control the computer or mouse. "
            f"CRITICAL: Do NOT use asterisks (*), markdown formatting, bold text, italics, or emojis in your final spoken response under ANY circumstances."
            f"{facts_block}"
            f"{tool_instructions}"
            f"{self_tooling_instructions}"
        )

    def warmup(self) -> None:
        """Send a tiny prompt to pre-load the model into VRAM."""
        info(f"Warming up model {self.model} into VRAM...")
        try:
            if OLLAMA_LIB_AVAILABLE:
                ollama_client.chat(model=self.model,
                                   messages=[{"role": "user", "content": "hi"}],
                                   keep_alive=-1,
                                   options={"num_predict": 1})
            else:
                data = _json.dumps({
                    "model": self.model,
                    "messages": [{"role": "user", "content": "hi"}],
                    "keep_alive": -1,
                    "stream": False,
                    "options": {"num_predict": 1}
                }).encode()
                req = urllib.request.Request(
                    "http://localhost:11434/api/chat",
                    data=data, headers={"Content-Type": "application/json"}
                )
                urllib.request.urlopen(req, timeout=30)
            success("Model warmed up and ready.")
        except Exception as e:
            warn(f"Warmup failed: {e}")

    def _trim_history(self) -> None:
        """Keep history within configured limits."""
        max_len = 1 + (CONVERSATION_HIST * 2)
        if len(self.history) > max_len:
            self.history = [self.history[0]] + self.history[-(max_len - 1):]

    def chat(self, user_input: str, on_think: Optional[Callable[[str], None]] = None) -> str:
        """Process user input: fetch module context, generate response, execute tools, store in history."""
        # Get direct module context (e.g. direct triggers)
        module_ctx = modules_registry.get_direct_context(user_input)
        if module_ctx:
            if "[SILENT_SUCCESS]" in module_ctx:
                clean_response = module_ctx.replace("[file_manager context]:", "").strip()
                self.history.append({"role": "user", "content": user_input})
                self.history.append({"role": "assistant", "content": clean_response})
                return clean_response
            augmented = f"{user_input}\n\n[System context — live module data:\n{module_ctx}]"
        else:
            augmented = user_input

        self.history.append({"role": "user", "content": augmented})
        self._trim_history()

        # Refresh system prompt (memory/tools may have changed)
        self.history[0] = {"role": "system", "content": self._get_system_prompt()}

        if OLLAMA_LIB_AVAILABLE:
            response_text = self._chat_lib(self.history)
        else:
            response_text = self._chat_http(self.history)

        if not response_text:
            response_text = "I'm sorry, I couldn't generate a response right now."

        # Extract <think> reasoning block
        think_match = re.search(r'<think>(.*?)</think>', response_text, re.DOTALL)
        if think_match:
            think_content = think_match.group(1).strip()
            print(c(f"\n[THINKING]: {think_content}\n", CYAN))
            if on_think:
                try:
                    on_think(think_content)
                except Exception:
                    pass
            # Clean think block out of actual response
            response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()

        # Check for autonomous LLM Tool Calling in response
        while True:
            module_name, params = extract_tool_call(response_text)
            if not module_name:
                break

            info(f"[TOOL CALL]: {module_name} with params: {_json.dumps(params)}")
            tool_result = modules_registry.execute_module(module_name, params, user_input=user_input)
            if tool_result is None:
                tool_result = f"Action '{module_name}' completed."
            else:
                tool_result = str(tool_result)

            if "[SILENT_SUCCESS]" in tool_result:
                response_text = tool_result
                break
            else:
                # Feed tool execution result back to LLM for final conversational summary or next step
                self.history.append({"role": "assistant", "content": response_text})
                self.history.append({"role": "user", "content": f"[Tool Result from {module_name}]:\n{tool_result}\nObserve the result. You may either execute another tool to continue your plan (e.g. TOOL_CALL: name {{args}}), or provide a final spoken response to the user."})

                if OLLAMA_LIB_AVAILABLE:
                    response_text = self._chat_lib(self.history)
                else:
                    response_text = self._chat_http(self.history)

                # Extract <think> reasoning block for follow-up response
                think_match = re.search(r'<think>(.*?)</think>', response_text, re.DOTALL)
                if think_match:
                    think_content = think_match.group(1).strip()
                    print(c(f"\n[THINKING]: {think_content}\n", CYAN))
                    if on_think:
                        try:
                            on_think(think_content)
                        except Exception:
                            pass

                # Clean any follow-up think block
                response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()

        self.history.append({"role": "assistant", "content": response_text})
        return response_text


    def _chat_lib(self, messages: List[Dict[str, str]]) -> str:
        """Chat using the official Ollama Python library."""
        try:
            response = ollama_client.chat(model=self.model, messages=messages, keep_alive=-1)
            content = response.get("message", {}).get("content", "")
            if not content:
                content = getattr(getattr(response, "message", None), "content", "")
            return content.strip()
        except Exception as e:
            error(f"Ollama library error: {e}")
            return ""

    def _chat_http(self, messages: List[Dict[str, str]]) -> str:
        """Chat using HTTP request to local Ollama API (fallback)."""
        try:
            data = _json.dumps({
                "model": self.model,
                "messages": messages,
                "keep_alive": -1,
                "stream": False
            }).encode()
            req = urllib.request.Request(
                "http://localhost:11434/api/chat",
                data=data, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = _json.loads(resp.read())
                return result.get("message", {}).get("content", "").strip()
        except Exception as e:
            error(f"Ollama HTTP error: {e}")
            return ""

    def check_available(self) -> bool:
        """Check if Ollama server is running."""
        try:
            urllib.request.urlopen("http://localhost:11434", timeout=3)
            return True
        except Exception:
            return False


# ─── Model selection ───────────────────────────────────────────────────────────

def list_ollama_models() -> List[str]:
    """Return list of locally installed Ollama model names."""
    try:
        if OLLAMA_LIB_AVAILABLE:
            result = ollama_client.list()
            models_raw = result.get("models", []) if isinstance(result, dict) \
                         else getattr(result, "models", [])
            return [m["name"] if isinstance(m, dict) else m.model for m in models_raw]
        else:
            with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5) as r:
                data = _json.loads(r.read())
                return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def select_ollama_model(explicit_model: Optional[str]) -> str:
    """If explicit_model is given, return it. Otherwise show interactive picker."""
    if explicit_model:
        return explicit_model

    models = list_ollama_models()

    if not models:
        warn("No Ollama models found or Ollama not running.")
        warn(f"Defaulting to '{DEFAULT_MODEL}'. Pull it with:  ollama pull {DEFAULT_MODEL}")
        return DEFAULT_MODEL

    if len(models) == 1:
        success(f"Using the only installed model: {c(models[0], GREEN)}")
        return models[0]

    # Interactive picker
    print(f"\n{BOLD}{CYAN}Select an Ollama model:{RESET}")
    print(c("─" * 44, DIM))
    for i, name in enumerate(models, 1):
        tag = c(f"[{i}]", CYAN)
        default_marker = c(" ◀ default", DIM) if name == DEFAULT_MODEL else ""
        print(f"  {tag} {name}{default_marker}")
    print(c("─" * 44, DIM))
    fallback = DEFAULT_MODEL if DEFAULT_MODEL in models else models[0]
    print(f"  {c('[Enter]', DIM)} use default ({fallback})")
    print()

    while True:
        try:
            raw = input(f"{BOLD}Your choice (1-{len(models)}): {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return fallback
        if not raw:
            return fallback
        if raw.isdigit() and 1 <= int(raw) <= len(models):
            chosen = models[int(raw) - 1]
            success(f"Selected model: {c(chosen, GREEN)}")
            return chosen
        print(c(f"  Please enter a number between 1 and {len(models)}", DIM))
