from llm.ollama_client import OllamaClient

class CustomerRAGAgent:
    def __init__(self, vector_store, transition_model):
        self.vs = vector_store
        self.tm = transition_model
        self.llm = OllamaClient()

    def run(self, history, context_text):
        try:
            last_item = history[-1]
            probs = self.tm.predict(last_item)
            evidence = self.vs.retrieve(context_text)

            prompt = f"""
            You are an explanation module in a retail AI system.

            IMPORTANT RULES:
            - You MUST explain only the items listed in "Predicted next-item probabilities".
            - You MUST NOT introduce new items.
            - You MUST NOT contradict the probabilities.
            - You MUST cite the retrieved evidence when explaining.

            Customer last purchased: {last_item}

            Predicted next-item probabilities (AUTHORITATIVE):
            {probs}

            Retrieved historical baskets (EVIDENCE):
            {evidence}

            TASK:
            Explain why the predicted next item(s) are likely purchases.
            """

            explanation = self.llm.generate(prompt)

            return {
                "last_item": last_item,
                "probabilities": probs,
                "evidence": evidence,
                "explanation": explanation
            }

        except Exception as e:
            return {
                "error": "Agent execution failed",
                "details": str(e)
            }
