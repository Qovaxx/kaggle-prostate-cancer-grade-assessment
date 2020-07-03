from src.psga.settings import PROCESSED_DIRPATH, KAGGLE_DATASET_NAME
from src.psga.data_source.reader import TIFFReader
from time import time
from tqdm import tqdm
import cv2
from src.psga.utils.pickle import save_pickle
import numpy as np


import matplotlib.pyplot as plt
def show(image):
    plt.figure()
    plt.imshow(image)
    plt.show()


reader = TIFFReader(PROCESSED_DIRPATH / KAGGLE_DATASET_NAME)


karolinska = list()
radboud = list()


for i in tqdm(range(reader.num_images), total=reader.num_images):
    record = reader.get(i, read_mask=False)
    image = cv2.resize(record.image, dsize=(0, 0), fy=1/16, fx=1/16)
    mean = np.mean(image, tuple(range(image.ndim-1)))
    std = np.std(image, tuple(range(image.ndim - 1)))

    if record.additional["data_provider"] == "karolinska":
        karolinska.append((mean, std))
    else:
        radboud.append((mean, std))

save_pickle(karolinska, "/data/raw/karolinska.pkl")
save_pickle(radboud, "/data/raw/radboud.pkl")
