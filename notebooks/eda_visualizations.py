import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure professional charting environment
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 15,
    'figure.titlesize': 16
})

RAW_DATA_PATH = "data/raw/application_train.csv"
OUTPUT_DIR = "reports/figures"

def run_analytical_eda():

    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Missing raw data asset at: {RAW_DATA_PATH}")
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Parsing primary data matrix columns...")
    cols = ['TARGET', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'AMT_CREDIT']
    df = pd.read_csv(RAW_DATA_PATH, usecols=cols)

    # DIAGNOSTIC 1: Target Class Imbalance
    print("Generating Diagnostic 1: Target Disproportions...")
    plt.figure(figsize=(6, 5))
    ax = sns.countplot(x='TARGET', data=df, palette='Set2')
    plt.title('Loan Default Target Balance (Skew Profile)')
    plt.xlabel('Target (0: Non-Default / 1: Default)')
    plt.ylabel('Total Application Volume')
    
    total = len(df)
    for p in ax.patches:
        percentage = f'{100 * p.get_height() / total:.2f}%'
        x_coord = p.get_x() + p.get_width() / 2 - 0.15
        y_coord = p.get_height() + (total * 0.01)
        ax.annotate(percentage, (x_coord, y_coord), fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/01_target_imbalance.png", dpi=300)
    plt.close()

    # DIAGNOSTIC 2: Age Distribution Dynamics vs Risk

    print("Generating Diagnostic 2: Demographic Age Shifts...")
    plt.figure(figsize=(10, 5))
    
    # Convert negative days representation into positive age variables
    df['AGE_YEARS'] = df['DAYS_BIRTH'] / -365.25
    
    sns.kdeplot(df[df['TARGET'] == 0]['AGE_YEARS'], label='Repaid (Target=0)', fill=True, color='#2ecc71', alpha=0.4)
    sns.kdeplot(df[df['TARGET'] == 1]['AGE_YEARS'], label='Defaulted (Target=1)', fill=True, color='#e74c3c', alpha=0.4)
    
    plt.title('Age Density Variance Across Credit Risk Profiles')
    plt.xlabel('Applicant Age (Years)')
    plt.ylabel('Probability Density')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/02_age_density_vs_target.png", dpi=300)
    plt.close()

    # DIAGNOSTIC 3: Anomaly Exposure (DAYS_EMPLOYED)

    print("Generating Diagnostic 3: Isolate Systemic Outliers...")
    
    # Isolate the legendary Home Credit anomaly: 365243 days (~1000 years of employment!)
    anomaly_count = (df['DAYS_EMPLOYED'] == 365243).sum()
    print(f"   ⚠️ Systemic Outlier Detected: found {anomaly_count:,} records containing '365243' in DAYS_EMPLOYED.")
    
    # Create an amended column for honest visualization
    df['DAYS_EMPLOYED_CLEAN'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)
    df['YEARS_EMPLOYED'] = df['DAYS_EMPLOYED_CLEAN'] / -365.25

    plt.figure(figsize=(10, 5))
    sns.histplot(data=df, x='YEARS_EMPLOYED', hue='TARGET', bins=40, kde=True, multiple="stack", palette='coolwarm')
    plt.title('Employment History Profile (Anomalous Blocks Purged)')
    plt.xlabel('Verified Employment Length (Years)')
    plt.ylabel('Application Density')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/03_employment_anomalies_purged.png", dpi=300)
    plt.close()

    print(f"\n✅ PHASE 1 COMPLETE: All 3 plots exported safely to '{OUTPUT_DIR}/'")
    print("Ready to execute Phase 2.")

if __name__ == "__main__":
    run_analytical_eda()