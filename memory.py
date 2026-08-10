import json
import os
import re
from typing import List

from utils import error, success

MEMORY_FILE = os.path.join(os.path.dirname(__file__), ".memory")

def load_memory() -> List[str]:
    """Load long-term memory facts from file."""
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            error(f"Failed to load memory: {e}")
    return []

def save_memory(facts: List[str]) -> None:
    """Save long-term memory facts to file."""
    try:
        with open(MEMORY_FILE, "w") as f:
            json.dump(facts, f, indent=4)
    except Exception as e:
        error(f"Failed to save memory: {e}")

def add_memory_fact(fact: str) -> bool:
    """Add a new fact to memory if it doesn't already exist."""
    facts = load_memory()
    if fact not in facts:
        facts.append(fact)
        save_memory(facts)
        success(f"Remembered: {fact}")
        return True
    return False

def auto_extract_facts(text: str) -> None:
    """Extract simple facts from text using regex and store them."""
    text_lower = text.lower()
    ignored_words = {'good', 'fine', 'ok', 'okay', 'great', 'nothing', 'none', 'well', 'bad'}
    
    # Regex patterns and corresponding fact templates
    patterns = [
        (r"(?:my name is|call me|i'm|i am) ([a-zA-Z]+)", "The user's name is {0}."),
        (r"(?:i live in|i am from|i'm from) ([a-zA-Z\s]+)", "The user lives in {0}."),
        (r"(?:my favorite|i love) ([a-zA-Z\s]+)", "The user loves {0}.")
    ]
    
    for pattern, fact_template in patterns:
        match = re.search(pattern, text_lower)
        if match:
            value = match.group(1).strip()
            # Simple filtering
            if value and value not in ignored_words and len(value) > 1:
                # Capitalize nicely for the fact
                value = value.title()
                fact = fact_template.format(value)
                add_memory_fact(fact)
