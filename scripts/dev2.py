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



image_transforms = Compose([HorizontalFlip(always_apply=True)])
crop_transforms = Compose([VerticalFlip(always_apply=True), Blur(blur_limit=30, always_apply=True), Normalize(always_apply=True), ToTensorV2()])


dataset = PSGATileMaskedClassificationDataset(path=str(PROCESSED_DIRPATH / KAGGLE_DATASET_NAME),
                                              image_transforms=image_transforms,
                                              crop_transforms=crop_transforms)


data_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=dataset.fast_collate_fn, num_workers=16)
for index, batch in enumerate(data_loader):
    print(f"index {index} \n")







    a = 4











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


