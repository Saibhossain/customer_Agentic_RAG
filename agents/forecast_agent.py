# agents/forecast_agent.py
class ForecastAgent:
    def cash_counter_forecast(self, traffic):
        if traffic > 200:
            return "Open 3 counters"
        return "Open 1 counter"
