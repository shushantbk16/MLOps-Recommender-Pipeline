import pandas as pd
import numpy as np
import os
import pickle
import time
import faiss
from sentence_transformers import SentenceTransformer
from lightfm import LightFM
from lightfm.data import Dataset
import pyarrow as pa
import pyarrow.parquet as pq

# --- 1. Define Paths ---
MODEL_DIR = "data/models"
PROCESSED_PATH = "data/processed"
BEHAVIORS_INPUT_PATH = os.path.join(PROCESSED_PATH, "behaviors_processed.parquet")
OUTPUT_FILE = os.path.join(PROCESSED_PATH, "ranker_training_data.parquet")

# --- THE SOLUTION ---
# We will only process 50 batches (50 * 100k = 5 million rows)
# This is more than enough and will save hours of time and disk space.
MAX_BATCHES = 50

def generate_features():
    """
    Loads all models and iterates through the TRAIN data to
    generate a feature set for the ranker, saving it to a single
    Parquet file in batches.
    """
    
    # --- Clean up any old, corrupted file ---
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
        print(f"DEBUG: Removed old/corrupted {OUTPUT_FILE}")
        
    # --- 2. Load ALL Prerequisite Models & Data ---
    print("DEBUG: Loading all models and mappings...")
    # SBERT/FAISS
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    faiss_index = faiss.read_index(os.path.join(MODEL_DIR, "articles.index"))
    df_id_map = pd.read_parquet(os.path.join(MODEL_DIR, "article_id_map.parquet"))
    news_to_faiss_idx = {news_id: idx for idx, news_id in enumerate(df_id_map['NewsID'])}
    all_article_embeddings = faiss_index.reconstruct_n(0, faiss_index.ntotal)
    print("DEBUG: SBERT/FAISS models loaded successfully.")

    # LightFM
    with open(os.path.join(MODEL_DIR, "lightfm_model.pkl"), "rb") as f:
        lightfm_model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "lightfm_dataset_map.pkl"), "rb") as f:
        lightfm_dataset = pickle.load(f)
    user_to_lightfm_idx = lightfm_dataset.mapping()[0]
    item_to_lightfm_idx = lightfm_dataset.mapping()[2]
    print("DEBUG: LightFM models loaded successfully.")

    # --- 3. Load Data for Training the Ranker ---
    print(f"DEBUG: Loading processed behaviors file: {BEHAVIORS_INPUT_PATH}")
    df_behaviors = pd.read_parquet(
        BEHAVIORS_INPUT_PATH,
        columns=['UserID', 'History', 'Impressions']
    )
    df_behaviors = df_behaviors.dropna(subset=['History', 'Impressions'])
    print(f"DEBUG: Loaded {len(df_behaviors)} impression logs.")

    # --- 4. Feature Engineering (Memory-Safe) ---
    print(f"DEBUG: Starting feature engineering... Will stop after {MAX_BATCHES} batches.")
    start_time = time.time()
    
    training_data_batch = []
    
    # SBERT helper function
    def get_sbert_score(history_items, candidate_item):
        try:
            history_indices = [news_to_faiss_idx[item] for item in history_items if item in news_to_faiss_idx]
            if not history_indices: return 0.0
            history_embeddings = all_article_embeddings[history_indices]
            
            candidate_idx = news_to_faiss_idx.get(candidate_item)
            if candidate_idx is None: return 0.0
            candidate_embedding = all_article_embeddings[candidate_idx].reshape(1, -1)
            
            dot_product = np.dot(history_embeddings, candidate_embedding.T)
            return np.max(dot_product)
        except Exception:
            return 0.0

    batch_num = 0
    total_rows_processed = 0
    writer = None 
    
    for row in df_behaviors.itertuples():
        user_id = row.UserID
        history_items = row.History.split(' ')
        
        lightfm_user_idx = user_to_lightfm_idx.get(user_id)
        if lightfm_user_idx is None: continue
            
        for impression in row.Impressions.split(' '):
            try:
                candidate_item, target = impression.split('-')
                target = int(target)
            except ValueError:
                continue
            
            lightfm_item_idx = item_to_lightfm_idx.get(candidate_item)
            if lightfm_item_idx is None: continue
            
            lightfm_score = lightfm_model.predict(lightfm_user_idx, [lightfm_item_idx])[0]
            sbert_score = get_sbert_score(history_items, candidate_item)
            
            training_data_batch.append({
                'lightfm_score': lightfm_score,
                'sbert_score': sbert_score,
                'target': target
            })
            total_rows_processed += 1
            
            if len(training_data_batch) >= 100000:
                df_batch = pd.DataFrame(training_data_batch)
                table = pa.Table.from_pandas(df_batch, preserve_index=False)
                
                if writer is None:
                    writer = pq.ParquetWriter(OUTPUT_FILE, table.schema)
                
                writer.write_table(table)
                
                training_data_batch = [] 
                batch_num += 1
                print(f"  ... DEBUG: Wrote batch {batch_num} (100k rows)")
                
                # --- THIS IS THE NEW CODE ---
                if batch_num >= MAX_BATCHES:
                    print(f"DEBUG: Reached MAX_BATCHES ({MAX_BATCHES}). Stopping loop.")
                    break # Exit the impression loop
        
        # This will exit the user loop
        if batch_num >= MAX_BATCHES:
            break

    # Write the final batch (if any)
    if training_data_batch:
        df_batch = pd.DataFrame(training_data_batch)
        table = pa.Table.from_pandas(df_batch, preserve_index=False)
        if writer is None: 
            writer = pq.ParquetWriter(OUTPUT_FILE, table.schema)
        writer.write_table(table)
        print(f"  ... DEBUG: Wrote final batch {batch_num+1} ({len(training_data_batch)} rows)")
    
    if writer:
        writer.close()
        print(f"\nFeature engineering complete in {time.time() - start_time:.2f}s")
        print(f"Total rows processed: {total_rows_processed}")
        print(f"Training data saved to: {OUTPUT_FILE}")
    else:
        print("!!!!!!!!!!!!!! ERROR: NO DATA PROCESSED. NO FILE WRITTEN. !!!!!!!!!!!!!!")

if __name__ == "__main__":
    generate_features()