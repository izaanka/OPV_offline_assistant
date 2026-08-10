"""Weather module for OPV Voice Assistant."""

import json
import urllib.request
import urllib.parse
from typing import Dict, Any

from modules_registry import BaseModule
from utils import warn


class WeatherModule(BaseModule):
    name = "weather"
    description = "Fetch current weather information. Parameters: {\"location\": \"city_name\"} (optional, default: \"auto\")."
    requires_confirmation = False

    def can_handle_direct(self, user_input: str) -> bool:
        lower = user_input.lower()
        return "weather" in lower or "temperature" in lower

    def parse_direct_args(self, user_input: str) -> Dict[str, Any]:
        return {"location": "auto"}

    def execute(self, params: Dict[str, Any], user_input: str = "") -> str:
        location = params.get("location", "auto") or "auto"
        try:
            url = f"https://wttr.in/{urllib.parse.quote(location)}?format=j1" if location != "auto" else "https://wttr.in/?format=j1"
            req = urllib.request.Request(url, headers={'User-Agent': 'curl/7.68.0'})
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode('utf-8'))
                current_condition = data['current_condition'][0]
                temp = current_condition['temp_C']
                desc = current_condition['weatherDesc'][0]['value']
                area = data['nearest_area'][0]['areaName'][0]['value']
                return f"Weather in {area}: {temp}°C, {desc}."
        except Exception as e:
            warn(f"Failed to fetch weather: {e}")
            return f"Failed to fetch weather for '{location}': {e}"
