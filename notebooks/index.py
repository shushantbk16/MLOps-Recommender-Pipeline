import pandas as pd

# Load the NEW, FAST parquet files!
df_news = pd.read_parquet("../data/processed/news_processed.parquet")
df_behaviors = pd.read_parquet("../data/processed/behaviors_processed.parquet")

# Look how fast this runs!
display(df_news.head())
display(df_behaviors.head())