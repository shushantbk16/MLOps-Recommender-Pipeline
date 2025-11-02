import pandas as pd
import os
import pickle

# --- 1. Define Paths ---
PROCESSED_PATH = "data/processed"
MODEL_DIR = "data/models"
BEHAVIORS_PATH = os.path.join(PROCESSED_PATH, "behaviors_processed.parquet")
OUTPUT_PATH = os.path.join(MODEL_DIR, "user_history_map.pkl")

def create_map():
    """
    Loads the large behaviors file *once* and saves
    the user_id -> history_list map to a small pickle file.
    """
    
    print(f"Loading {BEHAVIORS_PATH}... (This is the slow part)")
    df_behaviors = pd.read_parquet(BEHAVIORS_PATH, columns=['UserID', 'History'])
    df_behaviors = df_behaviors.dropna(subset=['History'])
    
    print("Creating history map dictionary...")
    user_history_map = df_behaviors.set_index('UserID')['History'].str.split(' ').to_dict()
    
    print(f"Saving map to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(user_history_map, f)
        
    print("Done. You can now run the Streamlit app.")

if __name__ == "__main__":
    create_map()