import ollama

class OllamaClient:
    def __init__(self, model="gemma3:1b"):
        self.model = model

    def generate(self, prompt):
        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"]
