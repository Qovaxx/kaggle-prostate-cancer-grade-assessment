from src.psga.settings import PROCESSED_DIRPATH, KAGGLE_DATASET_NAME
from src.psga.data_source.provider.kaggle import PSGAPatchSequenceClassificationDataset

import matplotlib.pyplot as plt
def show(image):
    plt.figure()
    plt.imshow(image)
    plt.show()


from albumentations import Compose, HorizontalFlip, Blur, Normalize, VerticalFlip
from albumentations.pytorch import ToTensorV2


image_transforms = Compose([HorizontalFlip(always_apply=True)])
crop_transforms = Compose([VerticalFlip(always_apply=True), Blur(blur_limit=30, always_apply=True), Normalize(always_apply=True), ToTensorV2()])

dataset = PSGAPatchSequenceClassificationDataset(PROCESSED_DIRPATH / KAGGLE_DATASET_NAME, image_transforms=image_transforms, crop_transforms=crop_transforms)
z = dataset[4544]


a = 4