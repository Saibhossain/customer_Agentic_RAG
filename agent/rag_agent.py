class CustomerRAGAgent:
    def __init__(self, vector_store, transition_model):
        self.vector_store = vector_store
        self.model = transition_model

    def decide_retrieval(self, history):
        return len(history) > 0  # simple decision logic

    def run(self, history):
        last_item = history[-1]

        retrieved = []
        if self.decide_retrieval(history):
            retrieved = self.vector_store.retrieve(
                " -> ".join(history)
            )

        probs = self.model.predict(last_item)

        explanation = {
            "last_item": last_item,
            "next_item_probabilities": probs,
            "retrieved_examples": retrieved
        }

        return explanation
