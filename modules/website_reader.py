"""Website reader module for OPV Voice Assistant — Fetches and parses webpage text."""

import re
import urllib.request
import urllib.parse
from typing import Dict, Any

from modules_registry import BaseModule
from utils import warn, info

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False


KNOWN_SITES = {
    "bbc": "https://www.bbc.com",
    "bbc news": "https://www.bbc.com/news",
    "cnn": "https://www.cnn.com",
    "nytimes": "https://www.nytimes.com",
    "new york times": "https://www.nytimes.com",
    "github": "https://github.com",
    "reddit": "https://www.reddit.com",
    "wikipedia": "https://www.wikipedia.org",
    "hacker news": "https://news.ycombinator.com",
    "hackernews": "https://news.ycombinator.com",
    "techcrunch": "https://techcrunch.com",
    "verge": "https://www.theverge.com",
    "the verge": "https://www.theverge.com",
    "google": "https://www.google.com"
}


class WebsiteReaderModule(BaseModule):
    name = "website_reader"
    description = (
        "Read and extract clean text content from a web page URL or known website. "
        "Parameters: {\"url\": \"https://example.com\" or site name like \"bbc\"}."
    )
    requires_confirmation = False

    def can_handle_direct(self, user_input: str) -> bool:
        lower = user_input.lower()
        if "http://" in lower or "https://" in lower or "www." in lower:
            return True
        if any(f".{t}" in lower for t in ["com", "org", "net", "io", "co.uk", "gov", "edu"]):
            return True
        triggers = [
            "read website", "read page", "fetch website", "open link", "summarize url",
            "website homepage", "on the website", "on the homepage", "website content",
            "homepage of", "frontpage of"
        ]
        if any(t in lower for t in triggers):
            return True
        return any(site in lower for site in KNOWN_SITES) and ("website" in lower or "homepage" in lower or "page" in lower or "site" in lower)

    def parse_direct_args(self, user_input: str) -> Dict[str, Any]:
        lower = user_input.lower()
        match = re.search(r'https?://[^\s]+', user_input)
        if match:
            return {"url": match.group(0)}
        match_www = re.search(r'www\.[^\s]+', user_input)
        if match_www:
            return {"url": "https://" + match_www.group(0)}

        # Check domain pattern like example.com
        match_domain = re.search(r'\b[a-zA-Z0-9-]+\.(?:com|org|net|io|co\.uk|gov|edu)\b', user_input)
        if match_domain:
            return {"url": "https://" + match_domain.group(0)}

        # Check known site lookup
        for site, url in KNOWN_SITES.items():
            if site in lower:
                return {"url": url}

        return {"url": ""}

    def execute(self, params: Dict[str, Any], user_input: str = "") -> str:
        url = params.get("url", "").strip()
        
        # If url is not given or not a full URL, attempt resolution
        if not url or not (url.startswith("http://") or url.startswith("https://")):
            extracted = self.parse_direct_args(user_input or url)
            if extracted.get("url"):
                url = extracted["url"]
            elif url.lower() in KNOWN_SITES:
                url = KNOWN_SITES[url.lower()]
            elif url and not url.startswith("http"):
                url = "https://www." + url + ".com"

        if not url or not (url.startswith("http://") or url.startswith("https://")):
            return "Error: No valid web URL could be resolved."

        info(f"Reading website: {url}")
        try:
            req = urllib.request.Request(
                url,
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) OPV-Assistant/1.0'}
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                raw_html = response.read().decode('utf-8', errors='ignore')

            if BS4_AVAILABLE:
                soup = BeautifulSoup(raw_html, 'html.parser')
                # Extract page title
                title = soup.title.string.strip() if soup.title and soup.title.string else url
                # Remove script and style elements
                for script in soup(["script", "style", "noscript", "svg"]):
                    script.extract()

                # Extract headings and main paragraphs specifically for rich structure
                headlines = []
                for h in soup.find_all(['h1', 'h2', 'h3', 'p', 'a']):
                    text_str = h.get_text().strip()
                    if len(text_str) > 15 and text_str not in headlines:
                        headlines.append(text_str)

                clean_text = "\n".join(headlines[:60]) if headlines else soup.get_text(separator=' ')
            else:
                title = url
                clean = re.sub(r'<(script|style).*?>.*?</\1>', '', raw_html, flags=re.DOTALL | re.IGNORECASE)
                clean_text = re.sub(r'<[^>]+>', ' ', clean)

            # Clean up whitespace
            lines = (line.strip() for line in clean_text.splitlines())
            clean_text = ' '.join(line for line in lines if line)

            # Limit context size to 2500 characters
            if len(clean_text) > 2500:
                clean_text = clean_text[:2500] + "... [content truncated]"

            if not clean_text:
                return f"Website {url} loaded but no main readable text was found."

            return f"Live content fetched from {url} ({title}):\n{clean_text}"

        except Exception as e:
            warn(f"Failed to read website {url}: {e}")
            return f"Error reading website {url}: {e}"
