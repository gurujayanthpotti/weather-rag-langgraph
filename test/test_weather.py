# tests/test_weather.py
import pytest
from weather import format_weather_summary

def test_format_weather_summary_basic():
    sample = {
        "name": "TestCity",
        "main": {"temp": 25, "feels_like": 27, "humidity": 60},
        "weather": [{"description": "light rain"}]
    }
    s = format_weather_summary(sample)
    assert "TestCity" in s
    assert "light rain" in s
    assert "25" in s
