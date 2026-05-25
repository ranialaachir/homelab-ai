import requests
from dotenv import load_dotenv
import os

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL")
MODEL = os.getenv("OLLAMA_MODEL")

def ask(prompt: str, system: str=None) -> str:
    """
    Send a prompt to the local Ollama instance and return the response text.
    Optionally accepts a system prompt to set the model's behaviour.
    """
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
    }
    if system:
        payload["system"] = system
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        return response.json()["response"]
