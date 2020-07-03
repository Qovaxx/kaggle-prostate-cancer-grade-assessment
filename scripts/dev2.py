from src.psga.settings import PROCESSED_DIRPATH, KAGGLE_DATASET_NAME
from src.psga.data_source.reader import TIFFReader
from time import time
from tqdm import tqdm
import cv2
from src.psga.utils.pickle import save_pickle
import numpy as np
from src.psga.grade import CancerGradeSystem


import matplotlib.pyplot as plt
def show(image):
    plt.figure()
    plt.imshow(image)
    plt.show()


reader = TIFFReader(PROCESSED_DIRPATH / KAGGLE_DATASET_NAME)
grader = CancerGradeSystem()

un = set([x["additional"]["gleason_score"] for x in reader.meta])

for i in range(reader.num_images):
    rec = reader.meta[i]
    isup = rec["label"]
    gleason = rec["additional"]["gleason_score"]
    if gleason == "negative":
        gl1 = 0; gl2 = 0
    else:
        gl1 = int(gleason.split("+")[0]); gl2 = int(gleason.split("+")[1])

    if isup != grader.gleason_to_isup(gl1, gl2):
        print(f"{rec['name']}: {isup} - {gleason}")






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