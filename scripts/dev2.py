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


import torch
labels = torch.full((1, 20), fill_value=5).long().squeeze()
labels2 = torch.ones((20), dtype=torch.long) * 5


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


# 12.583172082901001
# 5.107307195663452
# 12.33859395980835
# 7.231093645095825
# 1.6814360618591309
# 3.2577078342437744
# 2.139578104019165
# 1.2491989135742188
# 8.649527549743652
# 8.436623334884644










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


