import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from lightfm import LightFM
from lightfm.data import Dataset
import xgboost as xgb



# --- 1. Model Loading (Cached for Performance) --
@st.cache_resource
def load_all_models():
    """
    Loads all pre-computed models and mappings from disk.
    This is now much faster.
    """
    print("Loading all models... This will happen once.")
    models = {}
    MODEL_DIR = "data/models"
    
    # SBERT/FAISS
    models['sbert_model'] = SentenceTransformer('all-MiniLM-L6-v2')
    models['faiss_index'] = faiss.read_index(os.path.join(MODEL_DIR, "articles.index"))
    
    df_id_map = pd.read_parquet(os.path.join(MODEL_DIR, "article_id_map.parquet"))
    models['news_to_faiss_idx'] = {news_id: idx for idx, news_id in enumerate(df_id_map['NewsID'])}
    models['faiss_idx_to_news_id'] = {idx: news_id for news_id, idx in models['news_to_faiss_idx'].items()}
    models['all_article_embeddings'] = models['faiss_index'].reconstruct_n(0, models['faiss_index'].ntotal)
    models['article_text_map'] = df_id_map.set_index('NewsID')['text_feature'].to_dict()

    # LightFM
    with open(os.path.join(MODEL_DIR, "lightfm_model.pkl"), "rb") as f:
        models['lightfm_model'] = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "lightfm_dataset_map.pkl"), "rb") as f:
        lightfm_dataset = pickle.load(f)
        
    models['user_to_lightfm_idx'] = lightfm_dataset.mapping()[0]
    models['item_to_lightfm_idx'] = lightfm_dataset.mapping()[2] 
    models['lightfm_idx_to_news_id'] = {v: k for k, v in models['item_to_lightfm_idx'].items()}
    models['lightfm_num_items'] = models['lightfm_model'].item_embeddings.shape[0]

    # XGBoost Ranker
    with open(os.path.join(MODEL_DIR, "xgb_ranker.pkl"), "rb") as f:
        models['xgb_ranker'] = pickle.load(f)
        
    # --- THIS IS THE FIX ---
    # Load the pre-computed history map. This is now super fast!
    print("Loading pre-computed user history map...")
    HISTORY_MAP_PATH = os.path.join(MODEL_DIR, "user_history_map.pkl")
    with open(HISTORY_MAP_PATH, 'rb') as f:
        models['user_history_map'] = pickle.load(f)
    
    print("Models loaded successfully.")
    return models

# --- 2. Helper Functions (The Logic) ---
def get_sbert_score(models, history_items, candidate_item_id):
    try:
        candidate_idx = models['news_to_faiss_idx'].get(candidate_item_id)
        if candidate_idx is None: return 0.0
            
        history_indices = [models['news_to_faiss_idx'][item] for item in history_items if item in models['news_to_faiss_idx']]
        if not history_indices: return 0.0
            
        history_embeddings = models['all_article_embeddings'][history_indices]
        candidate_embedding = models['all_article_embeddings'][candidate_idx].reshape(1, -1)
        
        dot_product = np.dot(history_embeddings, candidate_embedding.T)
        return np.max(dot_product).item() 
    except Exception:
        return 0.0

def semantic_search(models, query, k=5):
    query_embedding = models['sbert_model'].encode([query]).astype('float32')
    D, I = models['faiss_index'].search(query_embedding, k)
    
    results = []
    for idx in I[0]:
        news_id = models['faiss_idx_to_news_id'].get(idx)
        text = models['article_text_map'].get(news_id, "Text not found.")
        results.append({'NewsID': news_id, 'Article Text': text[:200] + "..."})
    return pd.DataFrame(results)

def get_hybrid_recommendations(models, user_id, user_history, k=10):
    # STAGE 1: RETRIEVAL
    lightfm_score_map = {}
    lightfm_user_idx = models['user_to_lightfm_idx'].get(user_id)
    
    if lightfm_user_idx is not None:
        all_item_internal_indices = np.arange(models['lightfm_num_items'])
        scores = models['lightfm_model'].predict(lightfm_user_idx, all_item_internal_indices)
        candidate_internal_ids = np.argsort(-scores)[:100]
        
        for internal_id in candidate_internal_ids:
            news_id = models['lightfm_idx_to_news_id'].get(internal_id)
            if news_id:
                lightfm_score_map[news_id] = scores[internal_id].item()

    sbert_candidates = []
    if user_history:
        history_indices = [models['news_to_faiss_idx'][item] for item in user_history if item in models['news_to_faiss_idx']]
        if history_indices:
            history_embeddings = models['all_article_embeddings'][history_indices]
            avg_history_embedding = np.mean(history_embeddings, axis=0).reshape(1, -1).astype('float32')
            D, I = models['faiss_index'].search(avg_history_embedding, 100)
            sbert_candidates = [models['faiss_idx_to_news_id'].get(idx) for idx in I[0]]

    all_candidates = set(lightfm_score_map.keys()) | set(sbert_candidates)
    
    # STAGE 2: RANKING
    ranker_features = []
    for news_id in all_candidates:
        if not news_id: continue

        lightfm_score = lightfm_score_map.get(news_id, 0.0)
        sbert_score = get_sbert_score(models, user_history, news_id)
        
        ranker_features.append({
            'NewsID': news_id,
            'lightfm_score': lightfm_score, 
            'sbert_score': sbert_score
        })

    if not ranker_features:
        st.warning("No valid candidates found for this user.")
        return pd.DataFrame()

    df_rank = pd.DataFrame(ranker_features)
    X_rank = df_rank[['lightfm_score', 'sbert_score']]
    
    probabilities = models['xgb_ranker'].predict_proba(X_rank)[:, 1]
    df_rank['Final_Ranker_Score'] = probabilities
    
    df_ranked = df_rank.sort_values(by='Final_Ranker_Score', ascending=False)
    
    df_ranked['Article_Text'] = df_ranked['NewsID'].map(models['article_text_map'])
    df_ranked['Article_Text'] = df_ranked['Article_Text'].str.slice(0, 200) + "..."
    
    return df_ranked.head(k)

# --- 3. The Streamlit App UI ---

# Load models on first run
models = load_all_models()

st.set_page_config(layout="wide")
st.title("End-to-End MLOps Recommender System 🚀")
st.write("This app demonstrates a full two-stage hybrid recommendation pipeline, including solutions for the 'cold-start' problem.")

demo_articles = list(models['article_text_map'].items())[:20]

tab1, tab2, tab3 = st.tabs([
    "Demo 1: Item Cold-Start (Semantic Search)", 
    "Demo 2: User Cold-Start (New User)", 
    "Demo 3: Known User (Hybrid Recs)"
])

# --- DEMO 1 TAB ---
with tab1:
    st.header("Problem: A New Article is Published (Item Cold-Start)")
    st.markdown("""
    **How do you recommend a brand new article that no user has ever clicked on?**
    * **Solution: Content-Based Search (SBERT + FAISS).**
    
    This demo uses the **SBERT** model to convert a text query into a vector, then uses **FAISS** to find the most semantically similar articles.
    """)
    
    search_query = st.text_input("Enter a topic (e.g., 'global warming', 'new samsung phone'):", "new samsung phone", key="demo1_text")
    
    if st.button("Search", key="content_search"):
        with st.spinner("Finding similar articles..."):
            df_results = semantic_search(models, search_query, k=5)
            st.dataframe(df_results, use_container_width=True)

# --- DEMO 2 TAB ---
with tab2:
    st.header("Problem: A New User Arrives (User Cold-Start)")
    st.markdown("""
    **How do you personalize for a new user with no click history?**
    * **After their *first click***, we can start personalizing.
    
    This demo simulates a new user's experience. Select an article to "click" (this will be their *only* history).
    """)

    selected_article = st.selectbox(
        "Simulate your first click:", 
        options=demo_articles, 
        format_func=lambda x: f"{x[0]}: {x[1][:100]}...",
        key="demo2_selectbox"
    )
    
    if st.button("Get Recommendations", key="new_user_rec"):
        with st.spinner("Running two-stage pipeline for new user..."):
            new_user_id = "NEW_USER_DEMO"
            new_user_history = [selected_article[0]] # Get the NewsID
            
            df_hybrid_results = get_hybrid_recommendations(models, new_user_id, new_user_history, k=10)
            
            st.subheader("New User Recommendations")
            st.markdown("""
            **Notice the results below:**
            * `lightfm_score` is **0.0** (this new user is not in the LightFM model).
            * `sbert_score` is **non-zero**, based on the article you "clicked".
            * The **XGBoost Ranker** learned to trust the `sbert_score`!
            """)
            if not df_hybrid_results.empty:
                df_display = df_hybrid_results.rename(columns={
                    'lightfm_score': 'LightFM Score (Collab)',
                    'sbert_score': 'SBERT Score (Content)',
                    'Final_Ranker_Score': 'Final XGBoost Score'
                })
                df_display = df_display[['NewsID', 'Article_Text', 'Final XGBoost Score', 'SBERT Score (Content)', 'LightFM Score (Collab)']]
                
                st.dataframe(df_display.style.format(
                    {'LightFM Score (Collab)': '{:.2f}', 'SBERT Score (Content)': '{:.4f}', 'Final XGBoost Score': '{:.4f}'}
                ), use_container_width=True)
            else:
                st.error("An error occurred. No recommendations found.")

# --- DEMO 3 TAB ---
with tab3:
    st.header("Scenario: A Known User Returns (Full Hybrid Pipeline)")
    st.markdown("""
    **This is the full pipeline for a returning user with a rich history.**
    * The **LightFM** model provides collaborative candidates.
    * The **SBERT** model provides content-based candidates.
    * The **XGBoost Ranker** intelligently blends both scores.
    """)
    
    user_id_input = st.text_input("Enter a Known UserID (e.g., U13740, U555, U10000):", "U13740", key="demo3_text")
    
    if st.button("Get Recommendations", key="known_user_rec"):
        
        known_user_history = models['user_history_map'].get(user_id_input, [])
        with st.expander(f"Click to see User {user_id_input}'s Recent History (Sample)"):
            if known_user_history:
                history_sample_ids = known_user_history[-5:]
                history_data = []
                for news_id in history_sample_ids:
                    text = models['article_text_map'].get(news_id, "Article text not found.")
                    history_data.append({
                        "Clicked_NewsID": news_id,
                        "Article_Text": text[:150] + "..."
                    })
                df_history = pd.DataFrame(history_data)
                st.dataframe(df_history, use_container_width=True)
            else:
                st.info("This user has no click history in the dataset (but is known to LightFM).")

        with st.spinner("Running two-stage pipeline for known user..."):
            df_hybrid_results = get_hybrid_recommendations(models, user_id_input, known_user_history, k=10)
            
            st.subheader(f"Top 10 Recommendations for {user_id_input}")
            st.markdown("Notice. See how the ranker balances both scores based on the history above.")
            if not df_hybrid_results.empty:
                df_display = df_hybrid_results.rename(columns={
                    'lightfm_score': 'LightFM Score (Collab)',
                    'sbert_score': 'SBERT Score (Content)',
                    'Final_Ranker_Score': 'Final XGBoost Score'
                })
                df_display = df_display[['NewsID', 'Article_Text', 'Final XGBoost Score', 'SBERT Score (Content)', 'LightFM Score (Collab)']]

                st.dataframe(df_display.style.format(
                    {'LightFM Score (Collab)': '{:.2f}', 'SBERT Score (Content)': '{:.4f}', 'Final XGBoost Score': '{:.4f}'}
                ), use_container_width=True)
            else:
                st.error("An error occurred. No recommendations found.")