import os
import json
import re
from difflib import SequenceMatcher
from typing import Optional, Tuple, Dict, List, Any

# Optional imports
try:
    from g2p_en import G2p
    _g2p = G2p()
    G2P_AVAILABLE = True
except ImportError:
    G2P_AVAILABLE = False

try:
    from utils import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

ALIASES_FILE = os.path.join(os.path.dirname(__file__), ".aliases.json")
COMPLEX_NOUNS_FILE = os.path.join(os.path.dirname(__file__), ".complex_nouns.json")

DEFAULT_ALIASES = {
    "stella": "Stellar", "stelar": "Stellar", "teller": "Stellar", "stellar": "Stellar",
    "is on": "Izaan", "is an": "Izaan", "eyes on": "Izaan", "ezan": "Izaan", "izan": "Izaan",
    "daily": "Delhi", "dilli": "Delhi", "bomb bay": "Mumbai"
}

GLOBAL_CITIES = [
    "Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Ahmedabad", "Chennai", "Kolkata", "Surat", "Pune", "Jaipur",
    "Lucknow", "Kanpur", "Nagpur", "Indore", "Thane", "Bhopal", "Visakhapatnam", "Pimpri-Chinchwad", "Patna",
    "Vadodara", "Ghaziabad", "Ludhiana", "Agra", "Nashik", "Faridabad", "Meerut", "Rajkot", "Kalyan-Dombivli",
    "Vasai-Virar", "Varanasi", "Srinagar", "Aurangabad", "Dhanbad", "Amritsar", "Navi Mumbai", "Allahabad",
    "Howrah", "Ranchi", "Gwalior", "Jabalpur", "Coimbatore", "Vijayawada", "Jodhpur", "Madurai", "Raipur",
    "Kota", "Guwahati", "Chandigarh", "Solapur", "Hubli-Dharwad", "Mysore", "Tiruchirappalli", "Bareilly",
    "Aligarh", "Tiruppur", "Gurgaon", "Moradabad", "Jalandhar", "Bhubaneswar", "Salem", "Warangal", "Mira-Bhayandar",
    "Thiruvananthapuram", "Bhiwandi", "Saharanpur", "Guntur", "Amravati", "Bikaner", "Noida", "Jamshedpur",
    "Bhilai", "Cuttack", "Firozabad", "Kochi", "Bhavnagar", "Dehradun", "Durgapur", "Asansol", "Nanded",
    "Kolhapur", "Ajmer", "Gulbarga", "Jamnagar", "Ujjain", "Loni", "Siliguri", "Jhansi", "Ulhasnagar",
    "New York", "London", "Paris", "Tokyo", "Sydney", "Dubai", "Singapore", "Hong Kong", "Los Angeles", "Chicago",
    "Toronto", "Berlin", "Rome", "Madrid", "Seoul", "Beijing", "Shanghai", "Moscow", "Istanbul", "São Paulo"
]

CITY_LOWER_MAP = {c.lower(): c for c in GLOBAL_CITIES}

def soundex_code(word: str) -> str:
    """Computes the Soundex code for a given word."""
    if not word: return ""
    word = word.upper()
    soundex = word[0]
    dictionary = {"BFPV": "1", "CGJKQSXZ": "2", "DT": "3", "L": "4", "MN": "5", "R": "6", "AEIOUHWY": "."}
    for char in word[1:]:
        for key in dictionary.keys():
            if char in key:
                code = dictionary[key]
                if code != "." and code != soundex[-1]:
                    soundex += code
                break
    soundex = soundex.replace(".", "")
    soundex = soundex.ljust(4, "0")
    return soundex[:4]

def load_complex_nouns() -> Tuple[Dict[str, str], Dict[str, str]]:
    """Loads complex nouns from file and built-in names."""
    nouns = {}
    if os.path.exists(COMPLEX_NOUNS_FILE):
        try:
            with open(COMPLEX_NOUNS_FILE, 'r') as f:
                nouns = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load complex nouns: {e}")

    for city in GLOBAL_CITIES:
        nouns[city] = city
        
    lower_map = {k.lower(): v for k, v in nouns.items()}
    soundex_map = {soundex_code(k): v for k, v in nouns.items()}
    return lower_map, soundex_map

def load_aliases() -> Dict[str, str]:
    """Loads aliases from the aliases file and defaults."""
    aliases = DEFAULT_ALIASES.copy()
    if os.path.exists(ALIASES_FILE):
        try:
            with open(ALIASES_FILE, 'r') as f:
                aliases.update(json.load(f))
        except Exception as e:
            logger.error(f"Failed to load aliases: {e}")
    return {k.lower(): v for k, v in aliases.items()}

def save_alias(phrase: str, correction: str) -> None:
    """Saves a new alias to the aliases file."""
    aliases = load_aliases()
    aliases[phrase.lower()] = correction
    try:
        with open(ALIASES_FILE, 'w') as f:
            json.dump(aliases, f, indent=4)
    except Exception as e:
        logger.error(f"Failed to save alias: {e}")

def phoneme_similarity(word_a: str, word_b: str) -> float:
    """Calculates phoneme similarity using G2P if available, else SequenceMatcher."""
    if G2P_AVAILABLE:
        phonemes_a = "".join(_g2p(word_a))
        phonemes_b = "".join(_g2p(word_b))
        return SequenceMatcher(None, phonemes_a, phonemes_b).ratio()
    else:
        return SequenceMatcher(None, word_a.lower(), word_b.lower()).ratio()

class PhonemeIndex:
    """Pre-computes phonemes for a list of terms and provides matching."""
    def __init__(self, terms: List[str]):
        self.terms = terms
        self.phoneme_map = {}
        if G2P_AVAILABLE:
            for term in terms:
                self.phoneme_map[term] = "".join(_g2p(term))
                
    def find_match(self, word: str, threshold: float = 0.7) -> Optional[str]:
        best_match = None
        best_score = 0.0
        
        if G2P_AVAILABLE:
            word_phonemes = "".join(_g2p(word))
            for term, term_phonemes in self.phoneme_map.items():
                score = SequenceMatcher(None, word_phonemes, term_phonemes).ratio()
                if score > best_score:
                    best_score = score
                    best_match = term
        else:
            for term in self.terms:
                score = phoneme_similarity(word, term)
                if score > best_score:
                    best_score = score
                    best_match = term
                    
        if best_score >= threshold:
            return best_match
        return None

def normalize_utterance(text: str) -> str:
    """Full normalization pipeline for speech transcriptions."""
    if not text:
        return text

    # 1. Fix weather mishears
    text = re.sub(r"(?i)\bwhat's to further\b", "what's the weather", text)
    
    # 2. Apply phonetic aliases
    aliases = load_aliases()
    for phrase, correction in aliases.items():
        pattern = re.compile(rf"\b{re.escape(phrase)}\b", re.IGNORECASE)
        text = pattern.sub(correction, text)
        
    # 3. City fuzzy correction after prepositions
    words = text.split()
    prepositions = {"in", "at", "near", "for", "weather", "climate", "from"}
    
    for i, word in enumerate(words):
        if i > 0 and words[i-1].lower() in prepositions:
            w_lower = word.lower()
            if w_lower in CITY_LOWER_MAP:
                words[i] = CITY_LOWER_MAP[w_lower]
                
    text = " ".join(words)
    
    # 4. G2P/Soundex correction for complex nouns (words >= 4 chars)
    lower_map, soundex_map = load_complex_nouns()
    words = text.split()
    
    for i, word in enumerate(words):
        if len(word) >= 4 and not word.istitle():
            w_lower = word.lower()
            if w_lower in lower_map:
                words[i] = lower_map[w_lower]
            else:
                sndx = soundex_code(word)
                if sndx in soundex_map:
                    words[i] = soundex_map[sndx]
                    
    return " ".join(words)
