from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import os
import json
import uuid
from document_processing_service import DocumentProcessingService
from catalog_loader import load_catalog
from recommendation_engine import RecommendationEngine

app = FastAPI()
processor = DocumentProcessingService()
catalog = load_catalog()
recommender = RecommendationEngine(catalog)

user_history_path = "user_history.json"
recommendations_path = "recommendations.json"

# Load user history and recommendations
user_history = {}
if os.path.exists(user_history_path):
    with open(user_history_path, "r") as f:
        user_history = json.load(f)

recommendations = {}
if os.path.exists(recommendations_path):
    with open(recommendations_path, "r") as f:
        recommendations = json.load(f)

RECOMMENDATION_THRESHOLD = 3


@app.post("/analyseDoc")
async def analyse_document(user_id: str = Form(...), file: UploadFile = File(...)):
    file_path = f"temp_{uuid.uuid4()}.pdf"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    print(f"Processing file: {file.filename}")
    text, category, embedding = processor.process_document(file_path)
    os.remove(file_path)

    user_history.setdefault(user_id, []).append({
        "doc_id": file.filename,
        "category": category,
        "embedding": embedding.tolist()
    })

    with open(user_history_path, "w") as f:
        json.dump(user_history, f, indent=2)

    return {"message": "Processed", "category": category}


@app.get("/userHistory/{user_id}")
def get_user_history(user_id: str):
    if user_id not in user_history:
        return JSONResponse(status_code=404, content={"message": "User history not found"})
    history = user_history.get(user_id, [])
    print(f"User {user_id} history count: {len(history)}; doc_ids: {[h['doc_id'] for h in history]}")
    return {"user_id": user_id, "history": history}


@app.get("/catalog")
def get_catalog():
   print(f"Catalog doc_ids: {[doc['doc_id'] for doc in catalog]}; count: {len(catalog)}")
   return [{"doc_id": doc["doc_id"], "category": doc["category"]} for doc in catalog]


@app.get("/recommend/{user_id}")
def recommend(user_id: str):
    history = user_history.get(user_id, [])
    if len(history) < RECOMMENDATION_THRESHOLD:
        return {"message": "Not enough history for recommendation"}

    result = recommender.recommend_for_user(history)
    if "doc_id" in result:
        recommendations[user_id] = result
        with open(recommendations_path, "w") as f:
            json.dump(recommendations, f, indent=2)
    return result


@app.post("/addToCatalog")
async def add_to_catalog(github_json_url: str = Form(...)):
    import requests

    # Fetch the JSON array from the GitHub content API
    resp = requests.get(github_json_url)
    resp.raise_for_status()
    files_info = resp.json()

    new_entries = []
    for file_info in files_info:
        download_url = file_info.get("download_url")
        if not download_url:
            continue
        file_resp = requests.get(download_url)
        file_resp.raise_for_status()
        file_name = download_url.split("/")[-1]
        temp_path = f"temp_catalog_{uuid.uuid4()}_{file_name}"
        with open(temp_path, "wb") as f:
            f.write(file_resp.content)

        text, category, embedding = processor.process_document(temp_path)
        os.remove(temp_path)

        new_entry = {
            "doc_id": file_name,
            "category": category,
            "download_url": download_url,
            "embedding": embedding.tolist()
        }
        catalog.append({
            "doc_id": file_name,
            "category": category,
            "download_url": download_url,
            "embedding": embedding  # np.array for recommendation engine
        })
        new_entries.append(new_entry)

    # Persist to catalog_embeddings.json
    embeddings_ingestion_json = "catalog_folder_ingestion_embeddings.json"
    if os.path.exists(embeddings_ingestion_json):
        with open(embeddings_ingestion_json, "r") as f:
            catalog_json = json.load(f)
    else:
        catalog_json = []

    catalog_json.extend(new_entries)
    with open(embeddings_ingestion_json, "w") as f:
        json.dump(catalog_json, f, indent=2)

    return {"message": "Catalog updated", "added": [e["doc_id"] for e in new_entries]}
