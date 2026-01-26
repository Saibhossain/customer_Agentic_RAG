from ingestion.ingest_transactions import ingest_transactions
from ingestion.ingest_documents import ingest_pdfs
from embeddings.vector_store import VectorStore
from models.transition_model import TransitionModel
from agents.rag_agent import CustomerRAGAgent
import pandas as pd

class SystemController:
    def __init__(self):
        self.vs = VectorStore()
        self.tm = TransitionModel()
        self.agent = CustomerRAGAgent(self.vs, self.tm)

    def ingest_transactions(self, csv_path):
        ingest_transactions(csv_path)
        return "Transactions ingested successfully"

    def ingest_documents(self, files):
        ingest_pdfs(files)
        return "Documents ingested successfully"

    def train_model(self, csv_path):
        df = pd.read_csv(csv_path)
        sequences = (
            df.sort_values("item_sequence")
              .groupby("transaction_id")["item"]
              .apply(list)
              .tolist()
        )
        self.tm.train(sequences)
        return "Prediction model trained"

    def query_agent(self, history, context):
        return self.agent.run(history, context)

    def system_stats(self):
        return {
            "vectors": self.vs.index.ntotal,
            "model_trained": len(self.tm.probs) > 0
        }
