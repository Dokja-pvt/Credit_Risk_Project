import os
import pandas as pd

RAW_TRAIN_PATH = "data/raw/application_train.csv"
RAW_PREV_PATH = "data/raw/previous_application.csv"

def validate_datasets():

    errors = 0

    # 1. File existence validation
    for path in [RAW_TRAIN_PATH, RAW_PREV_PATH]:
        if not os.path.exists(path):
            print(f"❌ CRITICAL ERROR: Target file missing at {path}")
            errors += 1
        else:
            print(f"✔️ Located file: {path}")

    if errors > 0:
        print("\n❌ Validation Failed. Resolve missing file constraints.")
        return

    # 2. Primary application table validation
    print("\nValidating application_train.csv structural rules...")
    df_train = pd.read_csv(RAW_TRAIN_PATH, nrows=5)
    
    if 'TARGET' not in df_train.columns:
        print("❌ SCHEMA ERROR: 'TARGET' column missing from primary tracking structure.")
        errors += 1
    if 'SK_ID_CURR' not in df_train.columns:
        print("❌ SCHEMA ERROR: Primary key identifier 'SK_ID_CURR' missing.")
        errors += 1

    # 3. Auxiliary table validation
    print("Validating previous_application.csv structural rules...")
    df_prev = pd.read_csv(RAW_PREV_PATH, nrows=5)
    if 'SK_ID_CURR' not in df_prev.columns:
        print("❌ SCHEMA ERROR: Relational foreign key 'SK_ID_CURR' missing from historical tables.")
        errors += 1

    if errors == 0:
        print("\n✅ PHASE 0 VALIDATION SUCCESSFUL: Data streams match target specifications.")
        print("Ready for Phase 1 (Exploratory Data Analysis).")
    else:
        print(f"\n❌ Validation halted: Found {errors} structural discrepancies.")

if __name__ == "__main__":
    validate_datasets()