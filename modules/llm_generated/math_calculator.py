"""LLM-generated module: math_calculator — Basic calculator for simple math operations."""

import io
import sys
from typing import Dict, Any
from modules_registry import BaseModule


class MathCalculatorModule(BaseModule):
    name = "math_calculator"
    description = "Basic calculator for simple math operations. Parameters schema: {\"a\": \"float\", \"b\": \"float\"}"
    requires_confirmation = False

    def execute(self, params: Dict[str, Any], user_input: str = "") -> str:
        _old_stdout = sys.stdout
        _captured = io.StringIO()
        sys.stdout = _captured
        try:
            def multiply(a, b):
                return a * b

            def add(a, b):
                return a + b

            # Dynamic evaluation based on params:
            print(eval(f"{params.get("a")} * {params.get("b")}"))
            _out = _captured.getvalue().strip()
            return _out if _out else "Execution completed."
        finally:
            sys.stdout = _old_stdout
