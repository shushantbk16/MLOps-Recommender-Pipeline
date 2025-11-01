import pandas as pd
import os
import pickle
import mlflow
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# --- 1. Define Paths ---
PROCESSED_PATH = "data/processed"
MODEL_DIR = "data/models"
TRAINING_DATA_PATH = os.path.join(PROCESSED_PATH, "ranker_training_data.parquet")
RANKER_MODEL_PATH = os.path.join(MODEL_DIR, "xgb_ranker.pkl")

def train_ranker():
    """
    Loads the pre-generated feature set and trains the
    XGBoost ranking model.
    """
    
    # --- 2. Load Training Data ---
    print(f"Loading training data from {TRAINING_DATA_PATH}...")
    df_train = pd.read_parquet(TRAINING_DATA_PATH)
    print(f"Loaded {len(df_train)} training samples.")
    
    # --- 3. Train XGBoost Model ---
    print("Training XGBoost classifier...")
    
    X = df_train[['lightfm_score', 'sbert_score']]
    y = df_train['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # We use XGBClassifier to predict the *probability* of a click
    xgb_ranker = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )

    xgb_ranker.fit(X_train, y_train)

    # --- 4. Evaluate Model ---
    print("Evaluating model...")
    preds = xgb_ranker.predict_proba(X_test)[:, 1] # Get probability of '1'
    auc_score = roc_auc_score(y_test, preds)
    print(f"  Test AUC: {auc_score:.4f}")

    # --- 5. Save Artifacts (DVC + MLflow) ---
    print("Saving XGBoost model...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(RANKER_MODEL_PATH, 'wb') as f:
        pickle.dump(xgb_ranker, f)

    print("Logging model to MLflow...")
    mlflow.set_experiment("xgb_ranker")

    with mlflow.start_run() as run:
        mlflow.log_params({
            "n_estimators": 100,
            "learning_rate": 0.1,
            "objective": "binary:logistic"
        })
        mlflow.log_metric("test_auc", auc_score)
        mlflow.log_artifact(RANKER_MODEL_PATH)
        
    print("\nRanker Model training complete.")
    print(f"MLflow Run ID: {run.info.run_id}")

if __name__ == "__main__":
    train_ranker()