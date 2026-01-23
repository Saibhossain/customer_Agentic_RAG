import pandas as pd
from models.transition_model import TransitionModel
from embeddings.vector_store import VectorStore
from agent.rag_agent import CustomerRAGAgent

df = pd.read_csv("data/update_dataset11.csv")
#print(df)
df = df.sort_values(["customer_id","timestamp"])
print(df[["customer_id","item_sequence","item","category","price"]])

sequences = (
    df.groupby("customer_id")["item"].apply(list).tolist()
)

tm = TransitionModel()
tm.train(sequences)

vs = VectorStore()
vs.add(["-> ".join(seq) for seq in sequences])

agent = CustomerRAGAgent(vs,tm)

history = ["Bread","Milk"]
result = agent.run(history)
print(result)