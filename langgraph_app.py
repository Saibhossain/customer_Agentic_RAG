from langgraph.graph import StateGraph, END

from agents.state import AgentState
from agents.planner_node import planner_node
from agents.prediction_node import prediction_node
from agents.document_rag_nod import document_rag_node
from agents.generator_node import generator_node

from llm.ollama_client import OllamaClient
from models.markov_model import MarkovNextItemModel
from embeddings.vector_store import VectorStore

# Initialize tools
llm = OllamaClient(model="gemma3:1b")
markov_model = MarkovNextItemModel()
vector_store = VectorStore()

# Build graph
graph = StateGraph(AgentState)

graph.add_node("planner", lambda s: planner_node(s, llm))
graph.add_node("prediction", lambda s: prediction_node(s, markov_model))
graph.add_node("document", lambda s: document_rag_node(s, vector_store, llm))
graph.add_node("generator", lambda s: generator_node(s, llm))

graph.set_entry_point("planner")

# ✅ FIXED ROUTER
def route_after_planner(state):
    plan = state["plan"]

    if plan == "PREDICTION":
        return "prediction"

    if plan == "DOCUMENT":
        return "document"

    if plan == "PREDICTION+DOCUMENT":
        return "prediction"

    return "prediction"  # safe fallback

graph.add_conditional_edges(
    "planner",
    route_after_planner,
    {
        "prediction": "prediction",
        "document": "document"
    }
)

graph.add_edge("prediction", "document")
graph.add_edge("document", "generator")
graph.add_edge("generator", END)

agent = graph.compile()

# Visualize graph
png_bytes = agent.get_graph().draw_mermaid_png()

with open("agent_graph.png", "wb") as f:
    f.write(png_bytes)

print("Agent graph saved as agent_graph.png")

# Run agent
result = agent.invoke({
    "user_query": "What will this customer buy next and why?",
    "last_item": "milk"
})

print(result["final_answer"])
