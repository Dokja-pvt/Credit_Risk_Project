# Enterprise Credit Risk Decisioning System

An end-to-end Machine Learning Engineering (MLE) system designed to optimize loan underwriting and expand financial inclusion for unbanked populations. This repository implements a decoupled, production-grade pipeline that validates raw schemas, processes relational historical portfolios, trains an imbalance-aware LightGBM classifier, and calibrates probabilities into traditional 300–850 credit scores.

---

## 👔 Business Case & Dataset Source

Many applicants struggle to secure traditional credit due to thin credit histories, making them vulnerable to predatory lenders. This system extracts behavioral signals from fragmented tables to predict default risk (~8% baseline) and transforms abstract model decimals into compliant, explainable metrics for risk officers.

- **Dataset Source:** [Home Credit Default Risk (Kaggle)](https://www.kaggle.com/c/home-credit-default-risk/data)
- **Core Files:** `application_train.csv` (demographics & target labels) and `previous_application.csv` (historical credit interactions).

---

## 🛠️ Repository Blueprint

Credit_Risk_Project/
│
├── data/
│ ├── raw/ # Source CSV binaries from Kaggle (git-ignored)
│ └── processed/ # Pipeline data matrices and serialized models
│
├── notebooks/
│ └── eda_visualizations.py # Automated visual diagnostic reporting script
│
├── src/
│ ├── **init**.py # Package initialization and public API exposures
│ ├── initialize_project.py # Automatic directory tree configuration setup
│ ├── validate_data.py # Upstream structural data validation gate
│ ├── data_merger.py # Relational history processing logic
│ ├── pipeline.py # Core LightGBM training engine module
│ └── scorecard.py # Financial log-odds conversion logic
│
├── reports/
│ └── figures/ # Exported high-definition analysis plots
│
├── app.py # Streamlit deployment core application
├── requirements.txt # Production package dependency manager
└── .gitignore # Outlines protected files excluded from tracking

---

## 🛡️ Modular Pipeline Breakdown

- **Data Validation Gate (`src/validate_data.py`)** — Enforces structural schema rules and key constraints (`SK_ID_CURR`) to prevent downstream pipeline failures.
- **Diagnostic Profiling (`notebooks/eda_visualizations.py`)** — Evaluates class skews and cleanses the legacy `365243` days-employed encoding error by converting it to statistical nulls (`np.nan`).
- **Relational Aggregation (`src/data_merger.py`)** — Condenses over 1.6 million rows of fragmented historical portfolios into behavioral aggregations (max, mean, sum) without memory leaks.
- **Core Machine Learning (`src/pipeline.py`)** — Trains a hyperparameter-optimized LightGBM classifier using stratified sampling and balanced class weights to combat high target skew.
- **Financial Calibration (`src/scorecard.py`)** — Transforms raw probability outputs into industry-standard credit scores using commercial log-odds scaling equations.
- **Interactive Interface (`app.py`)** — Exposes the final underwriting model through an interactive Streamlit UI web app for real-time risk assessment.

---

## 🧮 Scorecard Scaling Mechanics

Raw model probabilities (p) are mapped onto a commercial 300–850 point range using the standard financial log-odds formula:

- **Odds** = (1 - p) / p
- **Score** = Offset + Factor \* ln(Odds)

### Risk Tiers & Actions:

- **Tier 1 (Prime):** Score >= 720 | Low risk. Automated approval authorized.
- **Tier 2 (Good):** Score 640 - 719 | Standard exposure. Normal underwriting review.
- **Tier 3 (Subprime):** Score 540 - 639 | Elevated risk indicators. Requires collateral.
- **Tier 4 (High Risk):** Score < 540 | Critical underwriting failure. Automated decline.

---

## 💻 Software Engineering Standards

- **PEP 484 Type Hinting:** Strict input and output type declarations are maintained across all functional modules to ensure reliable runtimes.
- **Structured Logging Hierarchy:** Replaces basic print statements with Python's standard `logging` library, tracking timestamps, log levels (`INFO`, `ERROR`), and script line numbers.
- **Exposed API Architecture:** Package designs use `src/__init__.py` to declare public method bounds through `__all__`, flattening structural imports.

---

## ⚡ Setup & Execution Guide

### 1. Workspace Initialization

# Install required production dependencies

uv pip install -r requirements.txt

# Run programmatic environment directory builder

python src/initialize_project.py

_Note: Download the source data from the Kaggle link above and drop your unzipped CSV files into `data/raw/`._

### 2. Run the Processing and Training Pipeline

python src/validate_data.py
python notebooks/eda_visualizations.py
python src/data_merger.py
python src/pipeline.py
python src/scorecard.py

### 3. Launch the Serving UI

streamlit run app.py

---

## 🤝 Acknowledgements

- **Home Credit Group:** For providing the comprehensive, real-world data landscape to tackle global financial inclusion challenges.
- **Kaggle Community:** For hosting the open evaluation platform and fostering collaboration on data science best practices.
- **Open Source Contributors:** Special thanks to the maintainers of `scikit-learn`, `LightGBM`, `Pandas`, and `Streamlit` for providing the core building blocks of this modern MLOps system.
