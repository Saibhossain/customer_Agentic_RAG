
from agents.planner_agent import PlannerAgent
from agents.explanation_agent import ExplanationAgent
from embeddings.vector_store import VectorStore
from models.transition_model import TransitionModel

planner = PlannerAgent()
explainer = ExplanationAgent()
vs = VectorStore()

query = "why should we stock more eggs?"

decision = planner.decide(query)
evidence = vs.retrieve(query)

print(explainer.explain({"Eggs": 0.78}, evidence))
