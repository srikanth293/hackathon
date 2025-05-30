import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class RecommendationEngine:
    def __init__(self, catalog):
        self.catalog = catalog

    def recommend_for_user(self, user_history):
        if len(user_history) < 3:
            return {"message": "Not enough user history to make recommendations"}

        try:
            user_vectors = np.array([entry["embedding"] for entry in user_history])
        except KeyError:
            return {"message": "User history missing embeddings"}

        avg_user_vector = np.mean(user_vectors, axis=0).reshape(1, -1)

        similarities = []
        for doc in self.catalog:
            catalog_vector = np.array(doc["embedding"]).reshape(1, -1)
            sim = cosine_similarity(avg_user_vector, catalog_vector)[0][0]
            similarities.append((doc["doc_id"], sim))  # Fix: using "doc_id"

        similarities.sort(key=lambda x: x[1], reverse=True)
        top = similarities[0] if similarities and similarities[0][1] > 0.3 else None
        print(f"doc_id: {top[0]}")
        for doc in self.catalog:
            print(f"doc_id: {doc['doc_id']}, download_url: {doc['download_url']}")

        if top:
            doc_link = next(doc["download_url"] for doc in self.catalog if doc["doc_id"] == top[0])
            return {"recommended_document": top[0], "score": round(top[1], 4), "download_url": doc_link}
        else:
            return {"message": "No suitable recommendation found"}
