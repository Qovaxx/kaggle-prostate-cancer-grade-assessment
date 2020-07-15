import os.path as osp

from src.psga.grade import CancerGradeSystem
from src.psga.settings import (
    ARTIFACTS_DIRPATH,
    PROCESSED_DIRPATH,
    KAGGLE_DATASET_NAME,
    OVERALL_MEAN,
    OVERALL_STD
)

# Experiment settings
__postfix = "fold0"
EXPERIMENT_NAME = osp.splitext(osp.basename(__file__))[0] + "_" + __postfix
WORK_DIR = str(ARTIFACTS_DIRPATH / EXPERIMENT_NAME)
TRAIN_FUNC = "train_tile_classifier"

# Pipeline settings
DIST_PARAMS = dict(backend="nccl")
DEVICE = "cuda"
DEBUG = True
DEBUG_TRAIN_SIZE = 1000
MAX_EPOCHS = 1


# Data settings
__data_type = "src.psga.data_source.provider.kaggle.PSGATileMaskedClassificationDataset"
__psga_dirpath = str(PROCESSED_DIRPATH / KAGGLE_DATASET_NAME)
__fold = 0
__microns_tile_size=231

DATA_LOADER = dict(
    batch_per_gpu=1,
    workers_per_gpu=0,
    pin_memory=False,
)

DATA = dict(
    train=dict(type=__data_type, path=__psga_dirpath, phase="train", fold=__fold, tiles_intersection=0.5, batch_size=5,
               micron_tile_size=__microns_tile_size, crop_emptiness_degree=0.9, label_binning=True),

    val=dict(type=__data_type, path=__psga_dirpath, phase="val", fold=__fold, tiles_intersection=0.0,
             micron_tile_size=__microns_tile_size, crop_emptiness_degree=0.95, label_binning=True),

    test=dict(type=__data_type, path=__psga_dirpath, phase="test", tiles_intersection=0.0,
             micron_tile_size=__microns_tile_size, crop_emptiness_degree=0.95, label_binning=True),
)

# Transforms settings
__pre_transforms = []
__post_transforms = [
    dict(type="Normalize", mean=OVERALL_MEAN, std=OVERALL_STD,
         max_pixel_value=255.0, always_apply=True, p=1.0),
    dict(type="ToTensorV2")
]

__crop_transforms = [
    dict(type="HorizontalFlip", always_apply=False, p=0.5),
    dict(type="VerticalFlip", always_apply=False, p=0.5),
]

TRANSFORMS = dict(
    train=dict(image_transforms=[], crop_transforms=__pre_transforms + __crop_transforms + __post_transforms),
    val=dict(image_transforms=[], crop_transforms=__pre_transforms + __post_transforms),
    test=dict(image_transforms=[], crop_transforms=__pre_transforms + __post_transforms),
)


__classes = len(CancerGradeSystem().isup_grades) - 1
MODEL = dict(type="timm.models.senet.seresnext50_32x4d", pretrained=False, num_classes=__classes)

OPTIMIZER = dict(type="torch.optim.Adam", lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

SCHEDULER = dict(type="torch.optim.lr_scheduler.ExponentialLR", gamma=0.95, last_epoch=-1)

LOSSES = dict(bce=dict(type="torch.nn.BCEWithLogitsLoss", reduction="mean", pos_weight=None))

METRICS = dict(qwk=dict(type="src.psga.train.evaluation.metric.QuadraticWeightedKappa", labels=None, sample_weight=None))

BATCH_PROCESSOR = dict()


# Hook settings
HOOKS = [
    dict(type="ModifiedProgressBarHook", bar_width=10),

    dict(type="ModifiedPytorchDPHook"),
    dict(type="OptimizerHook", name="base"),
]
