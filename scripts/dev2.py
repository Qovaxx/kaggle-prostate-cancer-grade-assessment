from src.psga.settings import PROCESSED_DIRPATH, KAGGLE_DATASET_NAME
from src.psga.data_source.provider.kaggle import PSGATileSequenceClassificationDataset, PSGATileMaskedClassificationDataset
from albumentations import Compose, HorizontalFlip, Blur, Normalize, VerticalFlip
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import numpy as np

import matplotlib.pyplot as plt
def show(image):
    plt.figure()
    plt.imshow(image)
    plt.show()

from collections import Counter, defaultdict
from numpy.random import choice
import random
from typing import Iterable, List

labels = [5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] + [10] * 30 + [332] * 2  + [111] * 11111
# count = 20





indices = balanced_subsample(labels, 20)
print(Counter(np.array(labels)[indices]))



a = 4






image_transforms = Compose([HorizontalFlip(always_apply=True)])
crop_transforms = Compose([VerticalFlip(always_apply=True), Blur(blur_limit=30, always_apply=True), Normalize(always_apply=True), ToTensorV2()])


dataset = PSGATileMaskedClassificationDataset(path=str(PROCESSED_DIRPATH / KAGGLE_DATASET_NAME),
                                              fold=0, phase="train",
                                              image_transforms=image_transforms,
                                              crop_transforms=crop_transforms)

from time import time

for i in range(len(dataset)):
    start = time()
    z = dataset[i]
    print(time() - start)








# data_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=dataset.fast_collate_fn, num_workers=16)
# for index, batch in enumerate(data_loader):
#     print(f"index {index} \n")










# from timm.models.senet import seresnext50_32x4d
# import torch
# from torchvision.models._utils import IntermediateLayerGetter


# model = seresnext50_32x4d()
# embedder = IntermediateLayerGetter(model, return_layers={"avg_pool": "embedding"})
# embedder.cuda()
#
# images = torch.rand((10, 3, 512, 512))
# images = images.cuda()
#
# embeddings = embedder(images)["embedding"].flatten(1)
