import faiss
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(
        self,
        dim=384,
        index_path="storage/faiss.index",
        meta_path="storage/metadata.pkl"
    ):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.dim = dim
        self.index_path = index_path
        self.meta_path = meta_path

        os.makedirs("storage", exist_ok=True)

        self.index = None
        self.metadata = []

        self._load_or_initialize()

    def _load_or_initialize(self):
        """
        Industry-safe loading:
        - Try load
        - If corrupted → rebuild
        """
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.meta_path, "rb") as f:
                    self.metadata = pickle.load(f)

                # Sanity check
                if self.index.ntotal != len(self.metadata):
                    raise ValueError("Index/metadata size mismatch")

            except Exception as e:
                print(f"[WARN] FAISS index load failed, rebuilding. Reason: {e}")
                self._reset()
        else:
            self._reset()

    def _reset(self):
        """
        Clean rebuild
        """
        self.index = faiss.IndexFlatL2(self.dim)
        self.metadata = []

        # Remove corrupted files if they exist
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.meta_path):
            os.remove(self.meta_path)

    def add(self, texts, metadata):
        embeddings = self.model.encode(texts).astype(np.float32)
        self.index.add(embeddings)
        self.metadata.extend(metadata)
        self._save()

    def retrieve(self, query, k=3):
        if self.index.ntotal == 0:
            return []

        k = min(k, self.index.ntotal)

        q_emb = self.model.encode([query]).astype(np.float32)
        _, indices = self.index.search(q_emb, k)

        results = []
        for idx in indices[0]:
            if idx != -1:
                results.append(self.metadata[idx])
        return results

    def _save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
