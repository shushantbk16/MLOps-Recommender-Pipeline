import uvicorn
import pandas as pd
import numpy as np
import os
import pickle
import faiss
from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer
from lightfm import LightFM
from lightfm.data import Dataset
import xgboost as xgb
from contextlib import asynccontextmanager

# --- 1. Global Variables & Model Storage ---
models = {}

# --- 2. Model Loading (The "Lifespan" Event) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Loads all models into memory when the API starts.
    """
    print("Loading models... This may take a moment.")
    
    # --- Define Paths ---
    MODEL_DIR = "data/models"
    PROCESSED_PATH = "data/processed"
    
    # --- Load SBERT/FAISS (Content Model) ---
    models['sbert_model'] = SentenceTransformer('all-MiniLM-L6-v2')
    models['faiss_index'] = faiss.read_index(os.path.join(MODEL_DIR, "articles.index"))
    
    df_id_map = pd.read_parquet(os.path.join(MODEL_DIR, "article_id_map.parquet"))
    models['news_to_faiss_idx'] = {news_id: idx for idx, news_id in enumerate(df_id_map['NewsID'])}
    models['faiss_idx_to_news_id'] = {idx: news_id for news_id, idx in models['news_to_faiss_idx'].items()}
    models['all_article_embeddings'] = models['faiss_index'].reconstruct_n(0, models['faiss_index'].ntotal)
    print("DEBUG: SBERT/FAISS models loaded.")

    # --- Load LightFM (Collaborative Model) ---
    with open(os.path.join(MODEL_DIR, "lightfm_model.pkl"), "rb") as f:
        models['lightfm_model'] = pickle.load(f) # Store model in dict
    with open(os.path.join(MODEL_DIR, "lightfm_dataset_map.pkl"), "rb") as f:
        lightfm_dataset = pickle.load(f) # Load dataset map
        
    # --- THIS IS THE FIX ---
    # We must reference the models and maps *after* they are loaded.
    models['user_to_lightfm_idx'] = lightfm_dataset.mapping()[0]
    models['item_to_lightfm_idx'] = lightfm_dataset.mapping()[2] 
    models['lightfm_idx_to_news_id'] = {v: k for k, v in models['item_to_lightfm_idx'].items()}
    
    # We access the model *from the dictionary*
    models['lightfm_num_items'] = models['lightfm_model'].item_embeddings.shape[0]
    print("DEBUG: LightFM models and maps loaded.")

    # --- Load XGBoost (Ranker Model) ---
    with open(os.path.join(MODEL_DIR, "xgb_ranker.pkl"), "rb") as f:
        models['xgb_ranker'] = pickle.load(f)
    print("DEBUG: XGBoost ranker loaded.")
        
    # --- Load User History for SBERT ---
    print("Loading user history map...")
    BEHAVIORS_PATH = os.path.join(PROCESSED_PATH, "behaviors_processed.parquet")
    df_behaviors = pd.read_parquet(BEHAVIORS_PATH, columns=['UserID', 'History'])
    df_behaviors = df_behaviors.dropna(subset=['History'])
    models['user_history_map'] = df_behaviors.set_index('UserID')['History'].str.split(' ').to_dict()
    
    print("Models loaded successfully.")
    
    yield
    
    print("Shutting down and clearing models...")
    models.clear()

# --- 3. Create the FastAPI App ---
app = FastAPI(
    title="MLOps Recommender API",
    description="Serves a two-stage hybrid recommendation system.",
    lifespan=lifespan
)

# --- 4. Helper Functions (The Logic) ---
def get_sbert_score(history_items: list, candidate_item_id: str):
    try:
        candidate_idx = models['news_to_faiss_idx'].get(candidate_item_id)
        if candidate_idx is None: return 0.0
            
        history_indices = [models['news_to_faiss_idx'][item] for item in history_items if item in models['news_to_faiss_idx']]
        if not history_indices: return 0.0
            
        history_embeddings = models['all_article_embeddings'][history_indices]
        candidate_embedding = models['all_article_embeddings'][candidate_idx].reshape(1, -_1)
        
        dot_product = np.dot(history_embeddings, candidate_embedding.T)
        return np.max(dot_product).item() 
    except Exception:
        return 0.0

# --- 5. The API Endpoint ---
@app.get("/recommend/{user_id}")
async def get_recommendations(user_id: str, k: int = 10):
    
    if user_id not in models['user_to_lightfm_idx']:
        raise HTTPException(
            status_code=404, 
            detail=f"User {user_id} not found in LightFM model."
        )
        
    user_history = models['user_history_map'].get(user_id, [])
        
    # --- STAGE 1: RETRIEVAL (Corrected Logic) ---
    lightfm_user_idx = models['user_to_lightfm_idx'][user_id]
    all_item_internal_indices = np.arange(models['lightfm_num_items'])
    
    scores = models['lightfm_model'].predict(lightfm_user_idx, all_item_internal_indices)
    candidate_internal_ids = np.argsort(-scores)[:100]
    candidate_scores = scores[candidate_internal_ids]
    
    candidate_news_ids = [
        models['lightfm_idx_to_news_id'].get(internal_id) for internal_id in candidate_internal_ids
    ]

    # --- STAGE 2: RANKING ---
    ranker_features = []
    
    for news_id, lightfm_score in zip(candidate_news_ids, candidate_scores):
        
        if not news_id:
            continue

        sbert_score = get_sbert_score(user_history, news_id)
        
        ranker_features.append({
            'news_id': news_id,
            'lightfm_score': lightfm_score.item(), 
            'sbert_score': sbert_score
        })

    if not ranker_features:
        return {"user_id": user_id, "recommendations": []}

    df_rank = pd.DataFrame(ranker_features)
    X_rank = df_rank[['lightfm_score', 'sbert_score']]
    
    probabilities = models['xgb_ranker'].predict_proba(X_rank)[:, 1]
    df_rank['score'] = probabilities
    
    df_ranked = df_rank.sort_values(by='score', ascending=False)
    top_k_recs = df_ranked.head(k)['news_id'].tolist()

    return {
        "user_id": user_id,
        "recommendations": top_k_recs,
        "top_ranked_score": df_ranked.head(1)['score'].values[0].item()
    }

# --- 6. Run the API (for local testing) ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True, app_dir="src/api")