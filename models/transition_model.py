from collections import defaultdict

class TransitionModel:
    def __init__(self):
        self.counts = defaultdict(lambda: defaultdict(int))
        self.probs = {}

    def train(self, sequences):
        for seq in sequences:
            for i in range(len(seq) - 1):
                self.counts[seq[i]][seq[i + 1]] += 1

        for item, nexts in self.counts.items():
            total = sum(nexts.values())
            self.probs[item] = {k: v / total for k, v in nexts.items()}

    def predict(self, item):
        return self.probs.get(item, {})
