import os
from pathlib import Path


PROJECT_DIRPATH = Path(os.environ.get("PROJECT_DIRPATH", "/project"))
RAW_DIRPATH = Path(os.environ.get("RAW_DIRPATH", "/data/raw"))
PROCESSED_DIRPATH = Path(os.environ.get("PROCESSED_DIRPATH", "/data/processed"))
ARTIFACTS_DIRPATH = Path(os.environ.get("ARTIFACTS_DIRPATH", "/artifacts"))

DUPLICATES_PATH = PROJECT_DIRPATH / "data" / "duplicates"
EMPTY_MASKS_PATH = PROJECT_DIRPATH / "data" / "empty_masks"
MASK_LABEL_MISMATCH_PATH = PROJECT_DIRPATH / "data" / "mask_label_mismatch"

KAGGLE_DATASET_NAME = "prostate-cancer-grade-assessment"
SUBMISSION_FILENAME = "submission.csv"


KAROLINSKA_MEAN = [234.94419566, 215.22865016, 230.5719532]
KAROLINSKA_STD = [36.74102766, 66.36234617, 41.08997511]
RADBOUD_MEAN = [238.58295463, 220.36507001, 228.21944953]
RADBOUD_STD = [28.53086804, 51.33759978, 39.79333803]
OVERALL_MEAN = [236.71301302, 217.72548731, 229.42839042]
OVERALL_STD = [32.75003156, 59.0587483, 40.4596739]
