import faiss
import pickle
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self, dim=384):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = faiss.IndexFlatL2(dim)
        self.texts = []

    def add(self, texts):
        embeddings = self.model.encode(texts)
        self.index.add(embeddings)
        self.texts.extend(texts)

    def search(self, query, k=5):
        q_emb = self.model.encode([query])
        _, idx = self.index.search(q_emb, k)
        return [self.texts[i] for i in idx[0]]

    def save(self, index_path, meta_path):
        faiss.write_index(self.index, index_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.texts, f)

    def load(self, index_path, meta_path):
        self.index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            self.texts = pickle.load(f)
