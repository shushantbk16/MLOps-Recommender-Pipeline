import pandas as pd
import os
import time

# --- Define File Paths ---
RAW_PATH = "data/raw/MINDsmall_train/MINDlarge_train"
PROCESSED_PATH = "data/processed"

NEWS_INPUT = os.path.join(RAW_PATH, "news.tsv")
BEHAVIORS_INPUT = os.path.join(RAW_PATH, "behaviors.tsv")

NEWS_OUTPUT = os.path.join(PROCESSED_PATH, "news_processed.parquet")
BEHAVIORS_OUTPUT = os.path.join(PROCESSED_PATH, "behaviors_processed.parquet")

def process_data():
    """
    Loads raw .tsv files, processes them, and saves them
    as efficient .parquet files.
    """
    
    # Ensure the output directory exists
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    
    # --- 1. Process News Data ---
    print("Processing news.tsv...")
    start_time = time.time()
    
    news_cols = [
        'NewsID', 'Category', 'SubCategory', 'Title', 'Abstract',
        'URL', 'TitleEntities', 'AbstractEntities'
    ]
    df_news = pd.read_csv(NEWS_INPUT, sep='\t', header=None, names=news_cols)
    
    # --- Feature Engineering ---
    # Fill missing abstracts with the title (a common_practice)
    df_news['Abstract'] = df_news['Abstract'].fillna(df_news['Title'])
    
    # Create the 'text_feature' we'll use for SBERT
    df_news['text_feature'] = df_news['Title'] + ". " + df_news['Abstract']
    
    # Drop columns we don't need to save space
    df_news = df_news.drop(columns=['URL', 'TitleEntities', 'AbstractEntities'])
    
    # Save to Parquet
    df_news.to_parquet(NEWS_OUTPUT, index=False)
    
    print(f"  Saved news_processed.parquet in {time.time() - start_time:.2f}s")
    
    
    # --- 2. Process Behaviors Data ---
    print("\nProcessing behaviors.tsv... (This is the slow one)")
    start_time = time.time()
    
    behaviors_cols = [
        'ImpressionID', 'UserID', 'Time', 'History', 'Impressions'
    ]
    
    # Here's the slow step. We only do this once.
    df_behaviors = pd.read_csv(
        BEHAVIORS_INPUT, sep='\t', header=None, names=behaviors_cols
    )
    
    # --- Feature Engineering ---
    # Fill empty click histories with an empty string
    df_behaviors['History'] = df_behaviors['History'].fillna('')
    
    # Save to Parquet
    df_behaviors.to_parquet(BEHAVIORS_OUTPUT, index=False)
    
    print(f"  Saved behaviors_processed.parquet in {time.time() - start_time:.2f}s")
    
    print("\nData processing complete.")

# This makes the script runnable from the command line
if __name__ == "__main__":
    process_data()