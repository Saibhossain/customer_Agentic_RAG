# agents/planner_agent.py
class PlannerAgent:
    def decide(self, query):
        if "forecast" in query:
            return "forecast"
        if "why" in query:
            return "explain"
        return "analytics"
