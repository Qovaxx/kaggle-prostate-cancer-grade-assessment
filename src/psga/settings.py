import os
from pathlib import Path

from typing_extensions import Final

PROJECT_DIRPATH: Final = Path(os.environ.get("PROJECT_DIRPATH", "/project"))
RAW_DIRPATH: Final = Path(os.environ.get("RAW_DIRPATH", "/data/raw"))
PROCESSED_DIRPATH: Final = Path(os.environ.get("PROCESSED_DIRPATH", "/data/processed"))
ARTIFACTS_DIRPATH: Final = Path(os.environ.get("ARTIFACTS_DIRPATH", "/artifacts"))

KAGGLE_DATASET_NAME: Final = "prostate-cancer-grade-assessment"
SUBMISSION_FILENAME: Final = "submission.csv"
