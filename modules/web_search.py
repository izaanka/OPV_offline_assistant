"""Web search module for OPV Voice Assistant."""

import json
import urllib.request
import urllib.parse
from typing import Dict, Any

from modules_registry import BaseModule
from utils import warn


class WebSearchModule(BaseModule):
    name = "web_search"
    description = "Search Wikipedia and DuckDuckGo for general web information. Parameters: {\"query\": \"search_term\"}."
    requires_confirmation = False

    def can_handle_direct(self, user_input: str) -> bool:
        lower = user_input.lower()
        search_triggers = ["search for", "look up", "who is", "what is", "where is", "how to"]
        return any(t in lower for t in search_triggers)

    def parse_direct_args(self, user_input: str) -> Dict[str, Any]:
        lower = user_input.lower()
        search_triggers = ["search for", "look up", "who is", "what is", "where is", "how to"]
        for trigger in search_triggers:
            if trigger in lower:
                query = user_input[lower.find(trigger) + len(trigger):].strip()
                if query:
                    return {"query": query}
        return {"query": user_input}

    def execute(self, params: Dict[str, Any], user_input: str = "") -> str:
        query = params.get("query", "") or user_input
        if not query:
            return "No search query provided."

        results = []
        # Try Wikipedia
        try:
            url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(query)}&utf8=&format=json"
            req = urllib.request.Request(url, headers={'User-Agent': 'OPV-Assistant/1.0'})
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode('utf-8'))
                search_results = data.get('query', {}).get('search', [])
                if search_results:
                    snippet = search_results[0].get('snippet', '').replace('<span class="searchmatch">', '').replace('</span>', '')
                    title = search_results[0].get('title', '')
                    results.append(f"Wikipedia ({title}): {snippet}")
        except Exception as e:
            warn(f"Wikipedia search failed: {e}")

        # Try DuckDuckGo Lite
        try:
            url = "https://html.duckduckgo.com/html/"
            data = urllib.parse.urlencode({'q': query}).encode('utf-8')
            req = urllib.request.Request(url, data=data, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Content-Type': 'application/x-www-form-urlencoded'
            })
            with urllib.request.urlopen(req, timeout=5) as response:
                html = response.read().decode('utf-8')
                if 'class="result__snippet' in html:
                    s_start = html.find('class="result__snippet')
                    s_start = html.find('>', s_start) + 1
                    s_end = html.find('</a>', s_start)
                    snippet = html[s_start:s_end].strip().replace('<b>', '').replace('</b>', '')
                    if snippet:
                        results.append(f"Web Search: {snippet}")
        except Exception as e:
            warn(f"DuckDuckGo search failed: {e}")

        if results:
            return "\n".join(results)
        return f"No online results found for query: '{query}'"
