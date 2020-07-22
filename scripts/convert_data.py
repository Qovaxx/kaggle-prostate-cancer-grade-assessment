from src.psga.data_source.write import TIFFWriter
from src.psga.data_source.provider.kaggle import PSGADataAdapter
from src.psga.settings import (
    RAW_DIRPATH,
    PROCESSED_DIRPATH,
    KAGGLE_DATASET_NAME
)

writer = TIFFWriter(str(PROCESSED_DIRPATH / KAGGLE_DATASET_NAME), quality=90, tile_size=512)
adapter = PSGADataAdapter(str(RAW_DIRPATH / KAGGLE_DATASET_NAME), writer, verbose=True)
adapter.convert(processes=5)
