from src.psga.settings import PROCESSED_DIRPATH, KAGGLE_DATASET_NAME
from src.psga.data_source.read import TIFFReader
from time import time
from tqdm import tqdm
import cv2
from src.psga.utils.inout import save_pickle, load_pickle
import numpy as np
from src.psga.grade import CancerGradeSystem
import zlib
import pickle
from src.psga.data_source.provider.kaggle import PSGAClassificationDataset



import matplotlib.pyplot as plt
def show(image):
    plt.figure()
    plt.imshow(image)
    plt.show()

dataset = PSGAClassificationDataset(PROCESSED_DIRPATH / KAGGLE_DATASET_NAME)
z = dataset[312]


a = 4