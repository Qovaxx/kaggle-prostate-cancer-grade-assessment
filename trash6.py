from src.psga.utils.pickle import load_pickle
from pathlib import Path

path = Path("/data/processed/prostate-cancer-grade-assessment/temp").iterdir()
meta = [load_pickle(str(x)) for x in path]

a = 4