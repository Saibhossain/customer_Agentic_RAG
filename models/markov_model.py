import pandas as pd
from collections import defaultdict

class MarkovNextItemModel:
    def __init__(self):
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        self.transition_probs = {}

    def fit(self, df: pd.DataFrame):
        """ DATASET must contain:customer_id, transaction_id, timestamp, item_sequence, item """
        df = df.copy()
        df["item"] = df["item"].str.strip().str.lower()
        df = df.sort_values(["customer_id", "timestamp", "transaction_id", "item_sequence"])
        for _, group in df.groupby("customer_id"):
            items = group["item"].tolist()
            for i in range(len(items) - 1):
                current_item = items[i]
                next_item = items[i + 1]
                self.transition_counts[current_item][next_item] += 1

        self._compute_probabilities()

    def _compute_probabilities(self):
        for item, next_items in self.transition_counts.items():
            total = sum(next_items.values())
            self.transition_probs[item] = {
                nxt: count / total
                for nxt, count in next_items.items()
            }

    def predict_next(self, item: str, top_k=5):
        item = item.strip().lower()
        probs = self.transition_probs.get(item, {})

        return dict(
            sorted(probs.items(), key=lambda x: x[1], reverse=True)[:top_k]
        )


# for testing markov model
# df = pd.read_csv("/Users/mdsaibhossain/code/python/customer_Agentic_RAG/data/update_dataset11.csv")
#
# model = MarkovNextItemModel()
# model.fit(df)
# print(model.predict_next("wipes"))
