import requests

class OllamaClient:
    def __init__(self, model="gemma3:1b"):
        self.url = "http://localhost:11434/api/generate"
        self.model = model

    def generate(self, prompt):
        response = requests.post(
            self.url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )
        response.raise_for_status()
        return response.json()["response"]
