"""LLM rescoring pass 2 + contextual bias builder for Whisper initial_prompt."""

import json
import urllib.request
from typing import List, Optional

from utils import info, warn
from config import load_config
from memory import load_memory
from normalizer import load_aliases, load_complex_nouns, GLOBAL_CITIES

try:
    import ollama as ollama_client
    OLLAMA_LIB_AVAILABLE = True
except ImportError:
    OLLAMA_LIB_AVAILABLE = False

# Common English words to skip during rescoring heuristic
COMMON_WORDS = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "how", "who", "where", "when", "why", "can", "could", "is", "are", "was",
    "were", "been", "being", "me", "him", "them", "us", "its", "your", "our",
    "just", "about", "so", "very", "really", "much", "well", "good", "yes",
    "no", "hello", "hi", "hey", "thanks", "thank", "please", "okay", "ok",
    "right", "tell", "know", "think", "want", "like", "need", "get", "got",
}


def build_context_bias(wake_word: str = "", assistant_name: str = "") -> str:
    """Build a contextual bias prompt string from all known proper nouns.

    This string is injected into Whisper's initial_prompt to bias the decoder
    toward recognizing specific names, cities, and terms with zero latency cost.
    """
    terms = set()

    # 1. Extract capitalized words from memory facts
    try:
        facts = load_memory()  # returns list of strings
        for fact in facts:
            for word in fact.split():
                clean = word.strip(".,!?'\"")
                if clean and clean[0].isupper() and len(clean) > 2:
                    terms.add(clean)
    except Exception:
        pass

    # 2. Add all alias correction targets
    try:
        aliases = load_aliases()
        for v in aliases.values():
            terms.add(v)
    except Exception:
        pass

    # 3. Add complex noun values
    try:
        noun_lower_map, _ = load_complex_nouns()
        for v in noun_lower_map.values():
            terms.add(v)
    except Exception:
        pass

    # 4. Add global cities
    for city in GLOBAL_CITIES:
        terms.add(city)

    # 5. Add assistant name and wake word
    if assistant_name:
        terms.add(assistant_name)
    if wake_word and wake_word != "hey":
        terms.add(wake_word.capitalize())

    try:
        cfg = load_config()
        if cfg.get("assistant_name"):
            terms.add(cfg["assistant_name"])
    except Exception:
        pass

    term_list = ", ".join(sorted(terms))
    if term_list:
        return f"The following names and places may appear: {term_list}."
    return ""


def needs_rescoring(text: str) -> bool:
    """Heuristic: does this transcript likely contain misheard proper nouns?"""
    if not text:
        return False
    words = text.split()
    if len(words) <= 3:
        return False
    # If all words are common English, no rescoring needed
    has_uncommon = any(w.lower() not in COMMON_WORDS for w in words if len(w) > 2)
    return has_uncommon


def llm_rescore(raw_transcript: str, known_terms: List[str],
                model: str = "llama3.2:3b") -> str:
    """Pass 2: Use local Ollama LLM to correct proper nouns in speech transcript.

    Args:
        raw_transcript: The raw STT output text.
        known_terms: List of known proper nouns to bias correction.
        model: Ollama model to use for rescoring.

    Returns:
        Corrected transcript, or original on failure.
    """
    if not raw_transcript:
        return raw_transcript

    terms_str = ", ".join(known_terms[:50])  # Limit context size
    prompt = (
        f"Fix any misheard proper nouns in this speech transcript. "
        f"Known names/places: {terms_str}. "
        f"Return ONLY the corrected transcript, nothing else.\n\n"
        f"Transcript: \"{raw_transcript}\""
    )

    try:
        if OLLAMA_LIB_AVAILABLE:
            response = ollama_client.generate(
                model=model,
                prompt=prompt,
                options={"num_predict": 100, "temperature": 0.1}
            )
            result = response.get("response", "").strip().strip('"')
            return result if result else raw_transcript
        else:
            data = json.dumps({
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 100}
            }).encode()
            req = urllib.request.Request(
                "http://localhost:11434/api/generate",
                data=data, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read())
                corrected = result.get("response", "").strip().strip('"')
                return corrected if corrected else raw_transcript
    except Exception as e:
        warn(f"LLM rescoring failed: {e}")
        return raw_transcript
