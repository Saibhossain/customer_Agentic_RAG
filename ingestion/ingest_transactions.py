from embeddings.feature_encoder import encode_basket
from embeddings.vector_store import VectorStore
import pandas as pd

def ingest_transactions(csv_path):
    df = pd.read_csv(csv_path)
    vs = VectorStore()

    texts, metadata = [], []

    for _, g in df.groupby("transaction_id"):
        rows = g.sort_values("item_sequence").to_dict("records")
        texts.append(encode_basket(rows))
        metadata.append({
            "type": "transaction",
            "basket": rows
        })

    vs.add(texts, metadata)
