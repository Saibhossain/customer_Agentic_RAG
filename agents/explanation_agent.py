
from llm.ollama_client import OllamaClient

class ExplanationAgent:
    def explain(self, probs, evidence):
        prompt = f"""
                    RULES:
                    - Do not change predictions
                    - Explain only given items in short 
                    - with politely 
                    
                    Predictions: {probs}
                    Evidence: {evidence}
                    
                    Explain decision clearly.
                    """
        return OllamaClient().generate(prompt)
