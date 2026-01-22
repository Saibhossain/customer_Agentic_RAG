from sentence_transformers import SentenceTransformer
import faiss

class VectorStore:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = faiss.IndexFlatL2(384)
        self.texts = []
    def add(self,texts):
        embeddings = self.model.encode(texts)
        self.index.add(embeddings)
        self.texts.extend(texts)

    def retrieve(self,query,k=3):
        q_emb =self.model.encode([query])
        _, indices = self.index.search(q_emb,k)
        return [self.texts[i] for i in indices[0]]
