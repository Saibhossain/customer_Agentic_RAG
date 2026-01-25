import pandas as pd
from embeddings.feature_encoder import encode_basket
from embeddings.vector_store import VectorStore
from models.transition_model import TransitionModel
from agent.rag_agent import CustomerRAGAgent

# Load data
df = pd.read_csv("data/update_dataset11.csv")

# Build baskets & sequences
baskets = []
sequences = []

for _, group in df.groupby("transaction_id"):
    rows = group.sort_values("item_sequence").to_dict("records")
    baskets.append(rows)
    sequences.append([r["item"] for r in rows])

# Train model
tm = TransitionModel()
tm.train(sequences)

# Build vector store
vs = VectorStore()

texts = []
metadata = []

for basket in baskets:
    texts.append(encode_basket(basket))
    metadata.append(basket)

vs.add(texts, metadata)

# Run agent
agent = CustomerRAGAgent(vs, tm)

context = texts[0]
result = agent.run(["Bread", "Milk"], context)
print(result)
