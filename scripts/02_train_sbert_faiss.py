import pandas as pd
import numpy as np
import faiss
import os
import time
import mlflow
from sentence_transformers import SentenceTransformer

# --- 1. Define Paths ---
PROCESSED_NEWS_PATH = "data/processed/news_processed.parquet"
MODEL_DIR = "data/models"
FAISS_INDEX_PATH = os.path.join(MODEL_DIR, "articles.index")
ID_MAP_PATH = os.path.join(MODEL_DIR, "article_id_map.parquet")
SBERT_MODEL_NAME = 'all-MiniLM-L6-v2' # A fast, high-quality model

def train_content_model():
    """
    Trains the content-based retrieval model.
    1. Loads news data.
    2. Encodes text with SBERT.
    3. Builds a FAISS index.
    4. Saves index, ID map, and logs to MLflow.
    """
    
    # Ensure model directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # --- 2. Load Data ---
    print("Loading processed news data...")
    df_news = pd.read_parquet(PROCESSED_NEWS_PATH)
    
    # We only need unique articles for our index
    df_articles = df_news[['NewsID', 'text_feature']].drop_duplicates().reset_index(drop=True)
    print(f"Loaded {len(df_articles)} unique articles.")
    
    # --- 3. Load SBERT Model ---
    print(f"Loading SBERT model: {SBERT_MODEL_NAME}...")
    model = SentenceTransformer(SBERT_MODEL_NAME)
    
    # --- 4. Generate Embeddings ---
    print("Generating embeddings for all articles...")
    print("This may take a few minutes...")
    start_time = time.time()
    
    # model.encode() creates the vector embeddings
    article_embeddings = model.encode(
        df_articles['text_feature'].tolist(), 
        show_progress_bar=True
    )
    
    print(f"Embeddings generated. Shape: {article_embeddings.shape}")
    print(f"Time taken: {time.time() - start_time:.2f}s")
    
    # --- 5. Build FAISS Index ---
    print("Building FAISS index...")
    
    # Get the dimension of our vectors (384 for this model)
    d = article_embeddings.shape[1]
    
    # We use 'IndexFlatL2' - a simple, fast index for exact L2 (Euclidean) distance
    index = faiss.IndexFlatL2(d)
    
    # Add our vectors to the index
    # FAISS requires float32 numpy arrays
    index.add(article_embeddings.astype('float32'))
    
    print(f"FAISS index built. Total vectors: {index.ntotal}")
    
    # --- 6. Save Artifacts (DVC) ---
    print("Saving FAISS index and ID mapping...")
    faiss.write_index(index, FAISS_INDEX_PATH)
    
    # We MUST save the NewsID mapping. The FAISS index only stores
    # the row number (0, 1, 2...). This file maps row '0' back to 'N55528'.
    df_articles.to_parquet(ID_MAP_PATH, index=False)
    
    # --- 7. Log to MLflow ---
    print("Logging model to MLflow...")
    mlflow.set_experiment("content_recommender")
    
    with mlflow.start_run() as run:
        mlflow.log_param("sbert_model", SBERT_MODEL_NAME)
        mlflow.log_param("num_articles", len(df_articles))
        mlflow.log_param("vector_dimension", d)
        
        # Log the artifacts. This tells MLflow where our saved models are.
        mlflow.log_artifact(FAISS_INDEX_PATH)
        mlflow.log_artifact(ID_MAP_PATH)
        
    print("\nContent Model training complete.")
    print(f"MLflow Run ID: {run.info.run_id}")

if __name__ == "__main__":
    train_content_model()