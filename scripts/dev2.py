from src.psga.settings import PROCESSED_DIRPATH, KAGGLE_DATASET_NAME
from src.psga.data_source.provider.kaggle import PSGATileSequenceClassificationDataset, PSGATileMaskedClassificationDataset

import matplotlib.pyplot as plt
def show(image):
    plt.figure()
    plt.imshow(image)
    plt.show()


from albumentations import Compose, HorizontalFlip, Blur, Normalize, VerticalFlip
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader


import cv2
import numpy as np
from src.psga.transforms.slicer import TilesSlicer

# image = cv2.imread("/data/processed/gradient-1000-colours.jpg")
# slicer = TilesSlicer(256, intersection=0.5, emptiness_degree=0.5, remove_empty_tiles=True)
# tiles = slicer(image)




# tile_size = 256
# intersection = 0.3
# step = int(tile_size * intersection)
#
#
# x_steps = np.arange(0, image.shape[1], step)
# y_steps = np.arange(0, image.shape[0], step)
#
# pad_w = x_steps[-1] + tile_size - image.shape[1]
# pad_h = y_steps[-1] + tile_size - image.shape[0]
#
# pad_width = [[pad_h // 2, pad_h - pad_h // 2], [pad_w // 2, pad_w - pad_w // 2], [0, 0]]
# image = np.pad(image, pad_width=pad_width, mode="constant", constant_values=255)
#
# tiles = list()
# for y in y_steps:
#     for x in x_steps:
#         tiles.append(image[y: y+tile_size, x: x+ tile_size])
# tiles = np.asarray(tiles)





a = 4



image_transforms = Compose([HorizontalFlip(always_apply=True)])
crop_transforms = Compose([VerticalFlip(always_apply=True), Blur(blur_limit=30, always_apply=True), Normalize(always_apply=True), ToTensorV2()])


# dataset = PSGATileMaskedClassificationDataset(path=str(PROCESSED_DIRPATH / KAGGLE_DATASET_NAME),
#                                               image_transforms=image_transforms,
#                                               crop_transforms=crop_transforms)



dataset = PSGATileSequenceClassificationDataset(path=str(PROCESSED_DIRPATH / KAGGLE_DATASET_NAME),
                                                image_transforms=image_transforms,
                                                crop_transforms=crop_transforms)
z = dataset[123]
# data_loader = DataLoader(dataset, batch_size=3, shuffle=True, collate_fn=dataset.fast_collate_fn)



















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


# embedder_batch_size = 100
#
# batch = [torch.rand(torch.randint(low=190, high=200, size=(1,)).item(), 3, 224, 224) for _ in range(5)]
# model = seresnext50_32x4d()
# embedder = IntermediateLayerGetter(model, return_layers={"avg_pool": "embedding"})
# embedder.cuda()
# embedder.eval()
#
# with torch.no_grad():
#     batch_embeddings = list()
#     for sample in batch:
#         embeddings = list()
#         chunks_count = torch.ceil(torch.tensor(sample.size(0) / embedder_batch_size)).int().item()
#         for chunk in torch.chunk(sample, chunks=chunks_count, dim=0):
#             chunk = chunk.cuda()
#             embeddings.extend(embedder(chunk)["embedding"].flatten(1).cpu())
#
#         batch_embeddings.append(torch.stack(embeddings))



a = 4


