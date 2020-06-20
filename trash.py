from src.psga.data import TIFFWriter
from src.psga.data import PSGADataAdapter


writer = TIFFWriter("/data/processed/prostate-cancer-grade-assessment")
adapter = PSGADataAdapter("/data/raw/prostate-cancer-grade-assessment", writer, verbose=True,
                          layer=0, crop_tissue_roi=True, processes=10)
adapter.convert()


a = 4




