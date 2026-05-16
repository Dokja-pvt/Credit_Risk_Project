import os
import pandas as pd
import numpy as np

RAW_TRAIN_PATH = "data/raw/application_train.csv"
RAW_PREV_PATH = "data/raw/previous_application.csv"
MERGED_OUTPUT_PATH = "data/processed/master_dataset.csv"

def run_feature_engineering():

    # 1. Verification and Loading
    if not os.path.exists(RAW_TRAIN_PATH) or not os.path.exists(RAW_PREV_PATH):
        raise FileNotFoundError("Required raw data assets missing inside 'data/raw/'.")

    print("Loading primary application records...")
    df_train = pd.read_csv(RAW_TRAIN_PATH)
    print(f"Primary Application Shape: {df_train.shape[0]:,} rows, {df_train.shape[1]} columns")

    print("Loading auxiliary credit history records...")
    df_prev = pd.read_csv(RAW_PREV_PATH)
    print(f"Historical Portfolio Shape: {df_prev.shape[0]:,} rows, {df_prev.shape[1]} columns")

    # 2. Treating Systemic Outliers (Discovered in Phase 1 EDA)
    print("\nPurging systemic anomalies from training metrics...")
    df_train['DAYS_EMPLOYED'] = df_train['DAYS_EMPLOYED'].replace(365243, np.nan)
    df_prev['DAYS_FIRST_DRAWING'] = df_prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan)
    df_prev['DAYS_FIRST_DUE'] = df_prev['DAYS_FIRST_DUE'].replace(365243, np.nan)
    df_prev['DAYS_LAST_DUE_1ST_VERSION'] = df_prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan)
    df_prev['DAYS_LAST_DUE'] = df_prev['DAYS_LAST_DUE'].replace(365243, np.nan)
    df_prev['DAYS_TERMINATION'] = df_prev['DAYS_TERMINATION'].replace(365243, np.nan)

    # 3. Aggregating Auxiliary Historical Records
    print("Executing historical relational transformations...")
    
    # Compute statistical primitives grouped by unique Customer ID (SK_ID_CURR)
    prev_agg = df_prev.groupby('SK_ID_CURR').agg({
        'AMT_APPLICATION': ['max', 'mean'],
        'AMT_CREDIT': ['max', 'mean', 'sum'],
        'AMT_DOWN_PAYMENT': ['max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
        'DAYS_DECISION': ['min', 'max']
    })

    # Flatten multi-level operational indices into single strings
    prev_agg.columns = ['_'.join(col).upper() for col in prev_agg.columns.values]
    prev_agg.reset_index(inplace=True)
    print(f"Aggregated Historical Matrix Shape: {prev_agg.shape[0]:,} clients, {prev_agg.shape[1] - 1} metrics generated.")

    # 4. Joint Inter-Table Coupling
    print("Stitching historical telemetry matrices to master dataframe...")
    master_df = df_train.merge(prev_agg, on='SK_ID_CURR', how='left')

    # 5. Output Management
    os.makedirs(os.path.dirname(MERGED_OUTPUT_PATH), exist_ok=True)
    print(f"Exporting compiled master dataset to storage...")
    master_df.to_csv(MERGED_OUTPUT_PATH, index=False)
    
    print(f"\n✅ PHASE 2 COMPLETE: Master dataset saved to '{MERGED_OUTPUT_PATH}'")
    print(f"Final Compiled Dimensions: {master_df.shape[0]:,} records, {master_df.shape[1]} features.")
    print("Ready to execute Phase 3 (Model Pipeline).")

if __name__ == "__main__":
    run_feature_engineering()