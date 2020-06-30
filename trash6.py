from src.psga.utils.pickle import load_pickle
from pathlib import Path
from skimage.io import MultiImage

import matplotlib.pyplot as plt
def show(image):
    plt.figure()
    plt.imshow(image)
    plt.show()



image = MultiImage("/data/raw/prostate-cancer-grade-assessment/train_label_masks/418e6b6e39af708710e1e497ca629ee8_mask.tiff")
z1 = image[0]
z2 = image[1]
z3 = image[2]

show(z3)

a = 4