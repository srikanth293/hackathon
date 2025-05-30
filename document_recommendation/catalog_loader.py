import json
import numpy as np


def load_catalog(path="catalog_folder_ingestion_embeddings.json"):
    with open(path, "r") as f:
        raw_catalog = json.load(f)

    catalog = []
    for entry in raw_catalog:
        # Flexible key handling
        doc_id = entry.get("doc_id") or entry.get("doc_id") or "unknown"
        category = entry.get("category", "unknown")
        vector = entry.get("vector") or entry.get("embedding")
        download_url = entry.get("download_url")

        if vector is None:
            continue  # skip if no vector present

        catalog.append({
            "doc_id": doc_id,
            "category": category,
            "download_url": download_url,
            "embedding": np.array(vector)
        })

    return catalog
