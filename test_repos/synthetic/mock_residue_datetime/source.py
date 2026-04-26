"""Greeting service based on time of day."""
from datetime import datetime

def greeting() -> str:
    hour = datetime.now().hour
    if hour < 12:
        return "Good morning"
    elif hour < 18:
        return "Good afternoon"
    else:
        return "Good evening"

def timestamp_label() -> str:
    return datetime.now().strftime("%Y-%m-%d")
