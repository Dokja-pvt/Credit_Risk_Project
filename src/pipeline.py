import os
import re
import pickle
import logging
import pandas as pd
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import lightgbm as lgb

logger = logging.getLogger(__name__)

MASTER_DATA_PATH: str = "data/processed/master_dataset.csv"
MODEL_OUTPUT_PATH: str = "data/processed/model.pkl"
FEATURES_OUTPUT_PATH: str = "data/processed/trained_features.pkl"

def run_model_pipeline(data_path: str = MASTER_DATA_PATH,
                       model_path: str = MODEL_OUTPUT_PATH,
                       features_path: str = FEATURES_OUTPUT_PATH) -> Tuple[lgb.LGBMClassifier, List[str]]:
    """
    Transforms master categorical features, handles JSON-incompatible syntax columns,
    and trains an imbalance-aware LightGBM model optimized via Stratified train-test partitioning.
    """
    logger.info("Initializing core model configuration and compilation loops.")

    if not os.path.exists(data_path):
        logger.error(f"Execution failed: Master dataset absent at {data_path}")
        raise FileNotFoundError("Target processed master dataset matrix missing.")

    df = pd.read_csv(data_path)
    y = df['TARGET']
    X = df.drop(columns=['TARGET', 'SK_ID_CURR'])

    logger.info("Vectorizing discrete string matrices via One-Hot Encoding.")
    X = pd.get_dummies(X, drop_first=True)

    logger.info("Sanitizing feature name characters to secure backend JSON safety rules.")
    X.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', str(col)) for col in X.columns]
    trained_features: List[str] = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info("Compiling LightGBM Classifier hyperparameter parameters.")
    model = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=31,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    logger.info("Fitting model parameters across stratified folds.")
    model.fit(X_train, y_train)

    y_pred_probs = model.predict_proba(X_test)[:, 1]
    y_pred_hard = model.predict(X_test)
    
    auc_score = roc_auc_score(y_test, y_pred_probs)
    logger.info(f"Model evaluation compiled successfully. Validation ROC-AUC Score: {auc_score:.4f}")
    logger.info(f"\nClassification Analysis Report:\n{classification_report(y_test, y_pred_hard)}")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(features_path, 'wb') as f:
        pickle.dump(trained_features, f)
        
    logger.info("Production model weight metrics successfully serialized and exported.")
    return model, trained_features

if __name__ == "__main__":
    run_model_pipeline()