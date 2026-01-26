
class AnalyticsAgent:
    def top_products(self, probabilities):
        return sorted(probabilities.items(), key=lambda x: x[1], reverse=True)

