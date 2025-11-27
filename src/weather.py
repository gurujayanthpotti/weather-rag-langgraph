# src/weather.py
import os
import requests
from typing import Dict

OPENWEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"

def fetch_weather(city: str, api_key: str = None) -> Dict:
    """
    Fetch current weather for `city` from OpenWeatherMap.
    Returns the JSON response (dict).
    """
    if api_key is None:
        api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        raise ValueError("OpenWeatherMap API key not provided")

    params = {"q": city, "appid": api_key, "units": "metric"}
    resp = requests.get(OPENWEATHER_URL, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()

def format_weather_summary(weather_json: Dict) -> str:
    """
    Build a human-friendly summary string from OpenWeatherMap JSON.
    """
    name = weather_json.get("name", "Unknown location")
    main = weather_json.get("main", {})
    weather_list = weather_json.get("weather", [])
    weather_desc = weather_list[0]["description"] if weather_list else "N/A"
    temp = main.get("temp")
    feels_like = main.get("feels_like")
    humidity = main.get("humidity")
    summary = (f"Weather in {name}: {weather_desc}. "
               f"Temperature: {temp}°C (feels like {feels_like}°C). "
               f"Humidity: {humidity}%")
    return summary
