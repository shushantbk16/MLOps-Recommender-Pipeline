import pandas as pd
import numpy as np
import os
import pickle
import time
import mlflow
from lightfm import LightFM
from lightfm.data import Dataset
from scipy.sparse import coo_matrix

# --- 1. Define Paths ---
PROCESSED_NEWS_PATH = "data/processed/news_processed.parquet"
PROCESSED_BEHAVIORS_PATH = "data/processed/behaviors_processed.parquet"
MODEL_DIR = "data/models"
LIGHTFM_MODEL_PATH = os.path.join(MODEL_DIR, "lightfm_model.pkl")
DATASET_MAP_PATH = os.path.join(MODEL_DIR, "lightfm_dataset_map.pkl")

def interaction_generator(df_behaviors):
    """
    A memory-efficient generator that yields (UserID, NewsID) tuples
    one at a time, without building a giant list.
    """
    print("Starting interaction generator...")
    # Using itertuples is faster than iterrows
    for row in df_behaviors.itertuples(index=False):
        # Yield from History
        if row.History:
            for item in row.History.split(' '):
                yield (row.UserID, item)
        
        # Yield from Impressions (clicks only)
        if row.Impressions:
            for item in row.Impressions.split(' '):
                if item.endswith('-1'):
                    yield (row.UserID, item[:-2])

def train_lightfm_model():
    """
    Trains the LightFM collaborative filtering model.
    """
    
    # --- 2. Load Data ---
    print("Loading processed news and behaviors data...")
    df_news = pd.read_parquet(PROCESSED_NEWS_PATH, columns=['NewsID'])
    all_known_items = df_news['NewsID'].unique()

    # We only need UserID, History, and Impressions
    df_behaviors = pd.read_parquet(
        PROCESSED_BEHAVIORS_PATH, 
        columns=['UserID', 'History', 'Impressions']
    )
    all_known_users = df_behaviors['UserID'].unique()
    print(f"Loaded {len(all_known_users)} users and {len(all_known_items)} items.")

    # --- 3. Build LightFM Dataset (Mappings) ---
    print("Building LightFM Dataset (fitting mappings)...")
    start_time = time.time()
    dataset = Dataset()
    dataset.fit(users=all_known_users, items=all_known_items)
    print(f"  Fit mappings in {time.time() - start_time:.2f}s")

    # --- 4. Build Interactions Matrix (Memory-Efficient) ---
    print("Building interactions matrix from generator...")
    print("(This will take a while, but won't crash)...")
    start_time = time.time()
    
    # This is the key change. We pass the GENERATOR directly.
    # LightFM will pull 77M items one by one and build the
    # sparse matrix internally without exploding RAM.
    (interactions, weights) = dataset.build_interactions(
        interaction_generator(df_behaviors)
    )
    
    print(f"  Built interactions matrix with shape: {interactions.shape}")
    print(f"  Data preparation took {time.time() - start_time:.2f}s")

    # --- 5. Train LightFM Model ---
    print("Training LightFM model...")
    start_time = time.time()

    model = LightFM(
        no_components=30,
        loss='warp',
        random_state=42
    )

    model.fit(
        interactions=interactions,
        epochs=10,
        num_threads=4 # This will still warn about OpenMP, it's fine
    )

    print(f"  Model training took {time.time() - start_time:.2f}s")

    # --- 6. Save Artifacts (DVC) ---
    print("Saving LightFM model and dataset mapping...")
    with open(LIGHTFM_MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
        
    with open(DATASET_MAP_PATH, 'wb') as f:
        pickle.dump(dataset, f)

    # --- 7. Log to MLflow ---
    print("Logging model to MLflow...")
    mlflow.set_experiment("collaborative_recommender")

    with mlflow.start_run() as run:
        mlflow.log_param("model_type", "LightFM")
        mlflow.log_param("loss_function", "warp")
        mlflow.log_param("no_components", 30)
        mlflow.log_param("epochs", 10)
        
        mlflow.log_artifact(LIGHTFM_MODEL_PATH)
        mlflow.log_artifact(DATASET_MAP_PATH)
    
    print("\nCollaborative Model training complete.")
    print(f"MLflow Run ID: {run.info.run_id}")

if __name__ == "__main__":
    train_lightfm_model()