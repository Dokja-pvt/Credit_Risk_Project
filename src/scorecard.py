import os
import pickle
import re
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

MASTER_DATA_PATH: str = "data/processed/master_dataset.csv"
MODEL_PATH: str = "data/processed/model.pkl"
FEATURES_PATH: str = "data/processed/trained_features.pkl"
OUTPUT_SCORECARD_PATH: str = "data/processed/customer_credit_scores.csv"

def probability_to_score(p: np.ndarray, base_score: int = 600, base_odds: int = 50, pdo: int = 20) -> np.ndarray:
    """Transforms continuous default probabilities into traditional credit scores using log-odds scaling."""
    p_clipped = np.clip(p, 0.0001, 0.9999)
    odds = (1 - p_clipped) / p_clipped
    factor = pdo / np.log(2)
    offset = base_score - (factor * np.log(base_odds))
    score = offset + (factor * np.log(odds))
    return np.clip(score, 300, 850).astype(int)

def run_scorecard_calibration() -> None:
    """Loads operational models, aligns test arrays, and maps metrics onto risk tiers."""
    logger.info("Initializing portfolio financial calibration sequence.")

    if not os.path.exists(MASTER_DATA_PATH) or not os.path.exists(MODEL_PATH):
        logger.error("Calibration failed: Required model artifacts are missing.")
        raise FileNotFoundError("Missing operational configuration metrics.")

    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(FEATURES_PATH, 'rb') as f:
        trained_features = pickle.load(f)

    df = pd.read_csv(MASTER_DATA_PATH)
    customer_ids = df['SK_ID_CURR']
    y_actual = df['TARGET']

    X = df.drop(columns=['TARGET', 'SK_ID_CURR'])
    X = pd.get_dummies(X, drop_first=True)
    X.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', str(col)) for col in X.columns]
    X = X.reindex(columns=trained_features, fill_value=0)

    probabilities = model.predict_proba(X)[:, 1]
    scores = probability_to_score(probabilities)

    ledger = pd.DataFrame({
        'SK_ID_CURR': customer_ids,
        'ACTUAL_TARGET': y_actual,
        'DEFAULT_PROBABILITY': np.round(probabilities, 4),
        'CREDIT_SCORE': scores
    })

    conditions = [
        (ledger['CREDIT_SCORE'] >= 720),
        (ledger['CREDIT_SCORE'] >= 640) & (ledger['CREDIT_SCORE'] < 720),
        (ledger['CREDIT_SCORE'] >= 540) & (ledger['CREDIT_SCORE'] < 640),
        (ledger['CREDIT_SCORE'] < 540)
    ]
    tiers = ['Tier 1 (Prime)', 'Tier 2 (Good)', 'Tier 3 (Subprime)', 'Tier 4 (High Risk)']
    ledger['RISK_TIER'] = np.select(conditions, tiers, default='Unknown')

    os.makedirs(os.path.dirname(OUTPUT_SCORECARD_PATH), exist_ok=True)
    ledger.to_csv(OUTPUT_SCORECARD_PATH, index=False)
    logger.info(f"Production decision ledger compiled and saved to: {OUTPUT_SCORECARD_PATH}")

if __name__ == "__main__":
    run_scorecard_calibration()