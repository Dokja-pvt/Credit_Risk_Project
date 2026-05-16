import os
import re
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import lightgbm as lgb
import warnings

# Suppress operational integration warnings
warnings.filterwarnings('ignore')

MASTER_DATA_PATH = "data/processed/master_dataset.csv"
MODEL_OUTPUT_PATH = "data/processed/model.pkl"
FEATURES_OUTPUT_PATH = "data/processed/trained_features.pkl"

def run_model_pipeline():

    if not os.path.exists(MASTER_DATA_PATH):
        raise FileNotFoundError(f"Compiled master matrix missing at {MASTER_DATA_PATH}. Run Phase 2 first.")

    print("Loading Master Dataset...")
    df = pd.read_csv(MASTER_DATA_PATH)
    print(f"Dataset Dimensions: {df.shape[0]:,} records, {df.shape[1]} columns")

    # 1. Feature-Target Isolation
    y = df['TARGET']
    # Drop IDs and targets to eliminate data leakage risks
    X = df.drop(columns=['TARGET', 'SK_ID_CURR'])

    # 2. Vectorized Categorical Transformation (One-Hot Encoding)
    print("Encoding categorical tracking structures...")
    X = pd.get_dummies(X, drop_first=True)

    # 3. Dynamic Feature Name Sanitization for LightGBM Backend Compliance
    print("Sanitizing feature names for JSON-parser compatibility...")
    X.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', str(col)) for col in X.columns]
    
    # Save feature names order for downstream Streamlit prediction matching
    trained_features = list(X.columns)

    # 4. Stratified Split (Preserves the 8% Minority Default Ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training Matrix Struct: {X_train.shape[0]:,} patterns")
    print(f"Testing Matrix Struct:  {X_test.shape[0]:,} patterns\n")

    # 5. Initialize Imbalance-Aware LightGBM Engine
    print("Initializing LightGBM Architecture...")
    model = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=31,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    # 6. Execute Fitting Pipeline
    print("Executing model training loops...")
    model.fit(X_train, y_train)
    print("Model compilation complete.\n")

    # 7. Verification and Diagnostics Evaluation
    print("--- Performance Evaluation Report ---")
    y_pred_probs = model.predict_proba(X_test)[:, 1]
    y_pred_hard = model.predict(X_test)

    # Compute ROC-AUC (Primary Metric for Credit Risk)
    roc_auc = roc_auc_score(y_test, y_pred_probs)
    print(f"Validation ROC-AUC Score: {roc_auc:.4f}")
    print("\nDetailed Classification Matrix:")
    print(classification_report(y_test, y_pred_hard))

    # 8. Serialize and Save Production Artifacts
    print("Serializing trained model components to disk...")
    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    
    with open(MODEL_OUTPUT_PATH, 'wb') as f:
        pickle.dump(model, f)
        
    with open(FEATURES_OUTPUT_PATH, 'wb') as f:
        pickle.dump(trained_features, f)

    print(f"✅ PHASE 3 COMPLETE: Model saved to '{MODEL_OUTPUT_PATH}'")
    print("Ready to execute Phase 4 (Financial Scorecard Calibration).")

if __name__ == "__main__":
    run_model_pipeline()