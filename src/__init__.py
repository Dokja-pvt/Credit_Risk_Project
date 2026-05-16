#Enterprise Credit Risk Decisioning System Source Package.


import logging
from typing import List

# Define package metadata structures
__version__: str = "1.0.0"
__author__: str = "Furqan Mohammad"

# Configure package-wide logging format baseline
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)
logger.info("Initializing Enterprise Credit Risk Core Source Package.")

# Explicitly declare public API exports to simplify import paths across modules
from src.data_merger import run_feature_engineering
from src.pipeline import run_model_pipeline
from src.scorecard import run_scorecard_calibration

__all__: List[str] = [
    "run_feature_engineering",
    "run_model_pipeline",
    "run_scorecard_calibration"
]