"""Self-tooling module for OPV Voice Assistant — gives LLM ability to create, test, and register tools."""

import os
import ast
import json
import importlib.util
import subprocess
from typing import Dict, Any, List

from modules_registry import BaseModule, confirm_action
from utils import info, warn, error, success

def scan_code_safety(code: str) -> List[str]:
    """Scan the AST of the provided Python code for potentially dangerous operations."""
    violations = []
    
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        violations.append(f"SyntaxError during parsing: {e}")
        return violations

    sensitive_paths = ["/etc/passwd", "/etc/shadow", "~/.ssh", "~/.gnupg", "/boot/", "/sys/", "/proc/"]
    dangerous_builtins = ["eval", "exec", "__import__", "compile", "globals", "locals", "setattr", "delattr", "getattr"]

    for node in ast.walk(tree):
        # 1. Check for dangerous function calls
        if isinstance(node, ast.Call):
            func_name = None
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                # We can't perfectly resolve modules without execution, but we check common patterns
                if isinstance(node.func.value, ast.Name):
                    full_name = f"{node.func.value.id}.{node.func.attr}"
                    
                    if full_name in ["os.system", "os.popen", "os.remove", "os.unlink", "os.rmdir", 
                                     "os.removedirs", "shutil.rmtree", "shutil.move"]:
                        violations.append(f"Dangerous system call used: {full_name}()")
                    
                    if full_name in ["subprocess.Popen", "subprocess.run", "subprocess.call"]:
                        # Check if shell=True is passed
                        for kw in node.keywords:
                            if kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                                violations.append(f"Subprocess called with shell=True: {full_name}()")

            if func_name in dangerous_builtins:
                violations.append(f"Dangerous built-in function used: {func_name}()")
                
        # 2. Check for string literals containing sensitive paths
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for path in sensitive_paths:
                if path in node.value:
                    violations.append(f"Sensitive path referenced in string literal: '{path}'")

        # 3. Check for dangerous imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in ["ctypes", "signal"]:
                    violations.append(f"Dangerous module imported: {alias.name}")
                    
        if isinstance(node, ast.ImportFrom):
            if node.module in ["ctypes", "signal"]:
                violations.append(f"Dangerous module imported: {node.module}")

    return violations

def extract_imports(code: str) -> tuple[str, str]:
    """Extract top-level imports from code and return (imports, remaining_code)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return "", code
        
    imports = []
    other_lines = []
    
    code_lines = code.split("\n")
    
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for i in range(node.lineno - 1, node.end_lineno):
                imports.append(code_lines[i])
        else:
            for i in range(node.lineno - 1, node.end_lineno):
                other_lines.append(code_lines[i])
                
    # Basic implementation: If we cannot perfectly reconstruct, we fallback to a simpler approach
    # Let's just find lines starting with 'import ' or 'from ' that are at 0 indentation
    
    import_lines = []
    remaining_lines = []
    for line in code_lines:
        stripped = line.strip()
        if (line.startswith("import ") or line.startswith("from ")) and not line.startswith(" "):
            import_lines.append(line)
        else:
            remaining_lines.append(line)
            
    return "\n".join(import_lines), "\n".join(remaining_lines)

class ToolCreatorModule(BaseModule):
    name = "tool_creator"
    description = (
        "Self-tooling engine. Exposes a single tool with an 'action' parameter.\n"
        "Actions:\n"
        "1. 'write_tool': Generates a BaseModule tool. Params: tool_name (str), description (str), code (str: execute method body), parameters (dict: JSON schema), requires_confirmation (bool, default False).\n"
        "2. 'read_file': Reads file contents. Params: file_path (str).\n"
        "3. 'run_shell': Runs shell command. Params: command (str).\n"
        "4. 'test_tool': Tests a generated tool. Params: tool_name (str), test_params (dict, default {}).\n"
        "5. 'register_tool': Dynamically loads tool. Params: tool_name (str)."
    )
    requires_confirmation = False

    def execute(self, params: Dict[str, Any], user_input: str = "") -> str:
        # Handle cases where params might be wrapped in 'arg'
        if "action" not in params and "arg" in params:
            arg_val = params["arg"]
            if isinstance(arg_val, str):
                arg_str = arg_val.strip()
                for cand in [arg_str, arg_str.replace(r'\"', '"')]:
                    if cand.startswith("{"):
                        try:
                            unwrapped = json.loads(cand, strict=False)
                            if isinstance(unwrapped, dict) and "action" in unwrapped:
                                params = unwrapped
                                break
                        except Exception:
                            try:
                                decoder = json.JSONDecoder(strict=False)
                                unwrapped, _ = decoder.raw_decode(cand)
                                if isinstance(unwrapped, dict) and "action" in unwrapped:
                                    params = unwrapped
                                    break
                            except Exception:
                                pass
            elif isinstance(arg_val, dict) and "action" in arg_val:
                params = arg_val

        action = params.get("action")
        
        if not action:
            return f"Error: No 'action' specified. Provided params: {params}"
            
        if action == "write_tool":
            return self._write_tool(params)
        elif action == "read_file":
            return self._read_file(params)
        elif action == "run_shell":
            return self._run_shell(params)
        elif action == "test_tool":
            return self._test_tool(params)
        elif action == "register_tool":
            return self._register_tool(params)
        else:
            return f"Error: Unknown action '{action}'."

    def _write_tool(self, params: Dict[str, Any]) -> str:
        tool_name = params.get("tool_name")
        description = params.get("description")
        code = params.get("code")
        parameters = params.get("parameters", {})
        requires_confirmation = params.get("requires_confirmation", False)

        if not all([tool_name, description, code]):
            return "Error: 'tool_name', 'description', and 'code' are required for 'write_tool'."

        violations = scan_code_safety(code)
        if violations:
            violation_str = "\n".join(f"- {v}" for v in violations)
            info(f"Safety violations found in generated tool '{tool_name}':\n{violation_str}")
            confirmed = confirm_action(f"Safety violations found in generated code for '{tool_name}'", violation_str)
            if not confirmed:
                return f"Tool creation blocked by user due to safety violations:\n{violation_str}"

        imports, method_body = extract_imports(code)
        
        # Indent method body by 12 spaces to fit inside _llm_logic
        indented_body = "\n".join(f"            {line}" if line.strip() else "" for line in method_body.split("\n"))
        
        # Generate ClassName
        class_name = "".join(word.capitalize() for word in tool_name.split("_")) + "Module"

        # Safely escape values for Python source code
        escaped_description = description.replace('\\', '\\\\').replace('"', '\\"')
        params_json = json.dumps(parameters).replace('\\', '\\\\').replace('"', '\\"')
        
        # Build imports block (skip if empty)
        imports_block = f"\n{imports}\n" if imports.strip() else ""

        template = f'''\"\"\"LLM-generated module: {tool_name} — {description}\"\"\"

import io
import sys
from typing import Dict, Any
from modules_registry import BaseModule
{imports_block}

class {class_name}(BaseModule):
    name = "{tool_name}"
    description = "{escaped_description} Parameters schema: {params_json}"
    requires_confirmation = {requires_confirmation}

    def execute(self, params: Dict[str, Any], user_input: str = "") -> str:
        def _llm_logic():
{indented_body}
        
        _old_stdout = sys.stdout
        _captured = io.StringIO()
        sys.stdout = _captured
        try:
            _ret_val = _llm_logic()
            _out = _captured.getvalue().strip()
            
            _parts = []
            if _out:
                _parts.append("STDOUT:\\n" + _out)
            if _ret_val is not None:
                _parts.append("RETURN:\\n" + str(_ret_val))
                
            if _parts:
                return "\\n\\n".join(_parts)
            else:
                return "Execution completed with no output."
        finally:
            sys.stdout = _old_stdout
'''
        base_dir = os.path.dirname(os.path.abspath(__file__))
        llm_dir = os.path.join(base_dir, "modules", "llm_generated")
        os.makedirs(llm_dir, exist_ok=True)
        
        # Create __init__.py if missing
        init_file = os.path.join(llm_dir, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, "w") as f:
                f.write("# LLM Generated Modules\n")

        file_path = os.path.join(llm_dir, f"{tool_name}.py")
        with open(file_path, "w") as f:
            f.write(template)

        success(f"Tool '{tool_name}' written to {file_path}")
        return f"Successfully wrote tool '{tool_name}' to {file_path}"

    def _read_file(self, params: Dict[str, Any]) -> str:
        file_path = params.get("file_path") or params.get("path") or params.get("file")
        if not file_path:
            return "Error: 'file_path' is required for 'read_file'."

        abs_path = os.path.abspath(os.path.expanduser(file_path))
        base_dir = os.path.dirname(os.path.abspath(__file__))
        home_dir = os.path.expanduser("~")
        
        # Check permissions
        if not (abs_path.startswith(base_dir) or abs_path.startswith("/tmp/") or abs_path.startswith(home_dir)):
            return f"Error: Cannot read file outside of allowed directories (project root, /tmp, or home). Path: {abs_path}"

        if not os.path.exists(abs_path):
            return f"Error: File '{abs_path}' not found."

        try:
            with open(abs_path, "r", encoding="utf-8") as f:
                content = f.read()
                if len(content) > 5000:
                    content = content[:5000] + "\n... [truncated]"
                return content
        except Exception as e:
            return f"Error reading file '{abs_path}': {e}"

    def _run_shell(self, params: Dict[str, Any]) -> str:
        cmd = params.get("command")
        if not cmd:
            return "Error: 'command' is required for 'run_shell'."

        confirmed = confirm_action(f"Run shell command?", cmd)
        if not confirmed:
            return f"Command execution '{cmd}' was CANCELED by the user."

        try:
            res = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )

            output = ""
            if res.stdout:
                output += f"STDOUT:\n{res.stdout.strip()}\n"
            if res.stderr:
                output += f"STDERR:\n{res.stderr.strip()}\n"
            
            if not output:
                output = f"Command completed with exit code {res.returncode} (no output)."
                
            if len(output) > 3000:
                output = output[:3000] + "\n... [truncated]"
                
            return output

        except subprocess.TimeoutExpired:
            return f"Error: Command '{cmd}' timed out after 30 seconds."
        except Exception as e:
            return f"Error executing command '{cmd}': {e}"

    def _test_tool(self, params: Dict[str, Any]) -> str:
        tool_name = params.get("tool_name")
        test_params = params.get("test_params", {})
        
        if not tool_name:
            return "Error: 'tool_name' is required for 'test_tool'."

        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, "modules", "llm_generated", f"{tool_name}.py")
        
        if not os.path.exists(file_path):
            return f"Error: Tool file not found at {file_path}"

        try:
            spec = importlib.util.spec_from_file_location(f"llm_generated_{tool_name}", file_path)
            if spec is None or spec.loader is None:
                return f"Error: Failed to load module spec for {file_path}"
                
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            
            # Find BaseModule subclass
            module_class = None
            for attr_name in dir(mod):
                attr = getattr(mod, attr_name)
                if isinstance(attr, type) and issubclass(attr, BaseModule) and attr is not BaseModule:
                    module_class = attr
                    break
                    
            if not module_class:
                return f"Error: No BaseModule subclass found in {file_path}"
                
            instance = module_class()
            result = instance.execute(test_params)
            return f"Test result for '{tool_name}':\n{result}"
            
        except Exception as e:
            import traceback
            return f"Error testing tool '{tool_name}':\n{traceback.format_exc()}"

    def _register_tool(self, params: Dict[str, Any]) -> str:
        tool_name = params.get("tool_name")
        if not tool_name:
            return "Error: 'tool_name' is required for 'register_tool'."
            
        try:
            from modules_registry import hot_reload_module
            return hot_reload_module(tool_name)
        except ImportError:
            return "Error: 'hot_reload_module' function not found in modules_registry.py."
        except Exception as e:
            return f"Error registering tool '{tool_name}': {e}"
