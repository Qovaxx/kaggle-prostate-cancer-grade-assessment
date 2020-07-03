import os
from pathlib import Path


PROJECT_DIRPATH = Path(os.environ.get("PROJECT_DIRPATH", "/project"))
RAW_DIRPATH = Path(os.environ.get("RAW_DIRPATH", "/data/raw"))
PROCESSED_DIRPATH = Path(os.environ.get("PROCESSED_DIRPATH", "/data/processed"))
ARTIFACTS_DIRPATH = Path(os.environ.get("ARTIFACTS_DIRPATH", "/artifacts"))

KAGGLE_DATASET_NAME = "prostate-cancer-grade-assessment"
SUBMISSION_FILENAME = "submission.csv"
