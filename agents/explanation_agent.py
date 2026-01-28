
class ExplanationAgent:
    def __init__(self, llm):
        self.llm = llm

    def explain_markov_prediction(self, customer_id, last_item, predictions):
        if not predictions:
            return "There is insufficient historical data to predict the next item."
        prob_text = "\n".join(
            [f"- {item}: {prob:.2f}" for item, prob in predictions.items()]
        )

        prompt = f"""
                    You are a retail analytics assistant.
                    Customer ID: {customer_id}
                    Last purchased item: {last_item}
                    
                    Predicted next-item probabilities (from a Markov model):
                    {prob_text}
                    
                    Explain in simple terms:
                    - why these items are predicted
                    - what the probabilities mean
                    - how a business user should interpret this result
                    
                    Keep the explanation short, clear, and non-technical.
                    """

        return self.llm.generate(prompt)

