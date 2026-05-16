import requests
from dotenv import load_dotenv
import os

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL")
MODEL = os.getenv("OLLAMA_MODEL")

def ask(prompt, system=None):
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


if __name__ == "__main__":
    print("=== Basic prompt ===")
    print(ask("What is the capital of Morocco?"))

    print("\n=== With system prompt ===")
    print(ask(
        prompt="What is the capital of Morocco?",
        system="You are a sarcastic geography teacher who answers in exactly one sentence."
    ))
