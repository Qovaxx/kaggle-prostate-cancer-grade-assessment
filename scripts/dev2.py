from src.psga.settings import PROCESSED_DIRPATH, KAGGLE_DATASET_NAME
from src.psga.data_source.reader import TIFFReader
from time import time
from tqdm import tqdm
import cv2
from src.psga.utils.pickle import save_pickle, load_pickle
import numpy as np
from src.psga.grade import CancerGradeSystem


import matplotlib.pyplot as plt
def show(image):
    plt.figure()
    plt.imshow(image)
    plt.show()


# with open("/project/data/duplicates", "r") as f:
#     data = f.readlines()
#
# clear = list()
# for line in data[1:]:
#     clear.append(line.replace("\n", "").split(","))
#
# from collections import defaultdict
# dict_ = defaultdict(list)
# for line in clear:
#     dict_[int(line[1])].append(line[0])
#
# import os
# from tifffile import imread
# from shutil import copyfile
# for k, v in tqdm(dict_.items(), total=len(dict_)):
#     folder = os.path.join("/data/processed/aa", str(k))
#     os.makedirs(folder, exist_ok=True)
#     for image in v:
#         rec = [x for x in reader.meta if x["name"] == image][0]
#         image_path = os.path.join(reader.visualizations_path, '/'.join(rec["visualization"].split("/")[1:]))
#         dst = os.path.join(folder, os.path.basename(image_path))
#         copyfile(image_path, dst)



a = 4