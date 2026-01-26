from embeddings.vector_store import VectorStore

def ingest_logs(log_file):
    vs = VectorStore()
    with open(log_file) as f:
        lines = f.readlines()

    texts = lines
    meta = [{"type": "system_log"} for _ in lines]
    vs.add(texts, meta)
