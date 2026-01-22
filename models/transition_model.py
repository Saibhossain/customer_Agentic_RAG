from collections import defaultdict

class TransitionModel:
    def __init__(self):
        self.transitions = defaultdict(lambda: defaultdict(int))
        self.probabilities = {}

    def train(self, sequences):
        for seq in sequences:
            for i in range(len(seq) - 1):
                self.transitions[seq[i]][seq[i+1]] += 1

        for item, next_items in self.transitions.items():
            total = sum(next_items.values())
            self.probabilities[item] = {
                k: v / total for k, v in next_items.items()
            }

    def predict(self, item):
        return self.probabilities.get(item, {})
