"""Modules Registry for OPV Voice Assistant — Dynamic Plugin Architecture."""

import os
import abc
import json
import pkgutil
import importlib
import importlib.util
from typing import Dict, Any, Optional

from utils import info, warn, error, success


class BaseModule(abc.ABC):
    """Abstract base class for assistant modules/plugins."""
    name: str = "base_module"
    description: str = "Base module description"
    requires_confirmation: bool = False

    def can_handle_direct(self, user_input: str) -> bool:
        """Return True if user_input directly matches this module's direct keyword/trigger."""
        return False

    def parse_direct_args(self, user_input: str) -> Dict[str, Any]:
        """Extract parameters from direct user speech/text."""
        return {}

    @abc.abstractmethod
    def execute(self, params: Dict[str, Any], user_input: str = "") -> str:
        """Execute module action and return string result for context or user."""
        pass


_REGISTERED_MODULES: Dict[str, BaseModule] = {}
_TTS_CALLBACK = None


def register_tts_callback(callback):
    """Register the assistant's configured TTS engine callback."""
    global _TTS_CALLBACK
    _TTS_CALLBACK = callback


def speak_tts(text: str) -> bool:
    """Speak text using the assistant's registered TTS voice engine."""
    if _TTS_CALLBACK:
        try:
            return _TTS_CALLBACK(text)
        except Exception as e:
            warn(f"Registered TTS speech error: {e}")
    return False


def confirm_action(action_description: str, details: str = "") -> bool:
    """Show a GUI dialog pop-up (Tkinter messagebox) asking for user confirmation.
    Falls back to CLI terminal prompt if GUI environment is not available."""
    warn(f"[CONFIRMATION REQUIRED]: {action_description}")
    if details:
        warn(f"Details: {details}")

    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        prompt_msg = f"{action_description}\n\nDetails:\n{details}\n\nDo you want to proceed?"
        result = messagebox.askyesno("Assistant Action Request", prompt_msg, parent=root)
        root.destroy()
        return result
    except Exception as e:
        # Fallback to console input if Tkinter / X11 fails
        print(f"\n[CONFIRMATION NEEDED]: {action_description}")
        if details:
            print(f"Details: {details}")
        try:
            resp = input("Allow action? (y/N): ").strip().lower()
            return resp in ("y", "yes")
        except (KeyboardInterrupt, EOFError):
            return False


def load_modules() -> Dict[str, BaseModule]:
    """Dynamically import and register all BaseModule subclasses found in the modules/ directory."""
    global _REGISTERED_MODULES
    _REGISTERED_MODULES.clear()

    base_dir = os.path.dirname(__file__)
    modules_dir = os.path.join(base_dir, "modules")
    
    if not os.path.exists(modules_dir):
        os.makedirs(modules_dir, exist_ok=True)

    init_file = os.path.join(modules_dir, "__init__.py")
    if not os.path.exists(init_file):
        with open(init_file, "w") as f:
            f.write("# OPV Modules Package\n")

    for _, module_name, is_pkg in pkgutil.iter_modules([modules_dir]):
        if is_pkg or module_name.startswith("_"):
            continue
        try:
            mod = importlib.import_module(f"modules.{module_name}")
            for attr_name in dir(mod):
                attr = getattr(mod, attr_name)
                if isinstance(attr, type) and issubclass(attr, BaseModule) and attr is not BaseModule:
                    instance = attr()
                    _REGISTERED_MODULES[instance.name] = instance
        except Exception as e:
            warn(f"Failed to load module 'modules.{module_name}': {e}")

    llm_generated_dir = os.path.join(modules_dir, "llm_generated")
    llm_modules_count = 0
    if os.path.exists(llm_generated_dir):
        for _, module_name, is_pkg in pkgutil.iter_modules([llm_generated_dir]):
            if is_pkg or module_name.startswith("_"):
                continue
            try:
                mod = importlib.import_module(f"modules.llm_generated.{module_name}")
                for attr_name in dir(mod):
                    attr = getattr(mod, attr_name)
                    if isinstance(attr, type) and issubclass(attr, BaseModule) and attr is not BaseModule:
                        instance = attr()
                        _REGISTERED_MODULES[instance.name] = instance
                        llm_modules_count += 1
            except Exception as e:
                warn(f"Failed to load module 'modules.llm_generated.{module_name}': {e}")
        info(f"Found {llm_modules_count} LLM-generated modules.")

    # Load the tool_creator module from the project root (self-tooling engine)
    try:
        tool_creator_path = os.path.join(base_dir, "tool_creator.py")
        if os.path.exists(tool_creator_path):
            import tool_creator
            for attr_name in dir(tool_creator):
                attr = getattr(tool_creator, attr_name)
                if isinstance(attr, type) and issubclass(attr, BaseModule) and attr is not BaseModule:
                    instance = attr()
                    _REGISTERED_MODULES[instance.name] = instance
    except Exception as e:
        warn(f"Failed to load tool_creator module: {e}")

    info(f"Loaded {len(_REGISTERED_MODULES)} modules: {', '.join(_REGISTERED_MODULES.keys())}")
    return _REGISTERED_MODULES


def hot_reload_module(tool_name: str) -> str:
    """Reloads or loads a specific module from the llm_generated directory dynamically.
    
    Args:
        tool_name (str): The name of the tool/module to load.
        
    Returns:
        str: Success or error message.
    """
    global _REGISTERED_MODULES
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, "modules", "llm_generated", f"{tool_name}.py")
    
    if not os.path.exists(file_path):
        err_msg = f"Error: File {file_path} does not exist."
        error(err_msg)
        return err_msg

    try:
        spec = importlib.util.spec_from_file_location(f"modules.llm_generated.{tool_name}", file_path)
        if spec is None or spec.loader is None:
            err_msg = f"Error: Could not load spec for {tool_name}"
            error(err_msg)
            return err_msg
            
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        
        found = False
        for attr_name in dir(mod):
            attr = getattr(mod, attr_name)
            if isinstance(attr, type) and issubclass(attr, BaseModule) and attr is not BaseModule:
                instance = attr()
                _REGISTERED_MODULES[instance.name] = instance
                found = True
                
        if not found:
            err_msg = f"Error: No BaseModule subclass found in {tool_name}"
            error(err_msg)
            return err_msg
            
        success_msg = f"Module '{tool_name}' registered successfully. It is now available as a tool."
        info(success_msg)
        success(success_msg)
        return success_msg
        
    except Exception as e:
        err_msg = f"Error hot reloading module '{tool_name}': {e}"
        error(err_msg)
        return err_msg


def get_registered_modules() -> Dict[str, BaseModule]:
    if not _REGISTERED_MODULES:
        load_modules()
    return _REGISTERED_MODULES


def get_direct_context(user_input: str) -> Optional[str]:
    """Check if any registered module matches user_input directly and return executed context."""
    modules = get_registered_modules()
    results = []

    for mod_name, mod in modules.items():
        try:
            if mod.can_handle_direct(user_input):
                args = mod.parse_direct_args(user_input)
                info(f"[DIRECT TRIGGER]: Module '{mod_name}' with args: {args}")
                if mod.requires_confirmation:
                    confirmed = confirm_action(f"Execute module '{mod_name}'?", json.dumps(args, indent=2))
                    if not confirmed:
                        results.append(f"[{mod_name} execution canceled by user]")
                        continue
                res = mod.execute(args, user_input=user_input)
                if res:
                    results.append(f"[{mod_name} context]:\n{res}")
        except Exception as e:
            warn(f"Error in direct module trigger {mod_name}: {e}")

    if results:
        return "\n\n".join(results)
    return None


def execute_module(module_name: str, params: Dict[str, Any], user_input: str = "") -> str:
    """Execute a module by name with given parameters."""
    modules = get_registered_modules()
    if module_name not in modules:
        return f"Error: Unknown module '{module_name}'. Available modules: {list(modules.keys())}"

    mod = modules[module_name]
    info(f"[TOOL EXECUTE]: Module '{module_name}' with params: {params}")
    if mod.requires_confirmation:
        confirmed = confirm_action(f"Execute action '{module_name}'?", json.dumps(params, indent=2))
        if not confirmed:
            return f"Action '{module_name}' was CANCELED by the user."

    try:
        res = mod.execute(params, user_input=user_input)
        if res is None:
            res = f"Action '{module_name}' completed."
        else:
            res = str(res)
        info(f"[TOOL RESULT]: {res}")
        return res
    except Exception as e:
        return f"Error executing module '{module_name}': {e}"


def get_system_prompt_tool_instructions() -> str:
    """Build dynamic system prompt section describing available tools."""
    modules = get_registered_modules()
    if not modules:
        return ""

    lines = [
        "\n\n--- AVAILABLE TOOLS & MODULES ---",
        "You have access to system capabilities. If you need to read a website, work with files, run shell commands, check weather, or search the web, output a tool call using this exact format:",
        "TOOL_CALL: <module_name> {\"param_name\": \"value\"}",
        "\nAvailable tool modules:"
    ]
    for name, mod in modules.items():
        confirm_tag = " (Requires user pop-up confirmation)" if mod.requires_confirmation else ""
        lines.append(f"- Module name: `{name}`{confirm_tag}")
        lines.append(f"  Description: {mod.description}")

    lines.append("\nNote: Output TOOL_CALL ONLY if an external action/data is needed. You may chain multiple TOOL_CALLs in sequence if a task requires it (e.g., creating and testing a tool). When all tools have been executed, summarize the final result for the user without additional TOOL_CALLs.")
    return "\n".join(lines)
