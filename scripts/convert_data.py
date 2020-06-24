from src.psga.data_source import TIFFWriter
from src.psga.data_source import PSGADataAdapter
from src.psga.settings import (
    RAW_DIRPATH,
    PROCESSED_DIRPATH,
    KAGGLE_DATASET_NAME
)

writer = TIFFWriter(str(PROCESSED_DIRPATH / KAGGLE_DATASET_NAME), quality=90, tile_size=512)
adapter = PSGADataAdapter(str(RAW_DIRPATH / KAGGLE_DATASET_NAME), writer,
                          verbose=True, layer=0, crop_nonwhite_roi=True)
# The number of processes is memory dependent.
# At zero layer, on average, one process consumes 20 gigabytes, at the peak up to 60
adapter.convert(processes=4)
