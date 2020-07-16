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
DIST_PARAMS = dict(backend="gloo")
DEVICE = "cuda"
DEBUG = False
DEBUG_TRAIN_SIZE = 100
MAX_EPOCHS = 99999


# Data settings
__data_type = "src.psga.data_source.provider.kaggle.PSGATileMaskedClassificationDataset"
__psga_dirpath = str(PROCESSED_DIRPATH / KAGGLE_DATASET_NAME)
__fold = 0
__microns_tile_size=231

DATA_LOADER = dict(
    batch_per_gpu=1,
    train_workers_per_gpu=0, # 5
    val_workers_per_gpu=0, # 2
    pin_memory=False,
)

DATA = dict(
    train=dict(type=__data_type, path=__psga_dirpath, phase="train", fold=__fold, tiles_intersection=0.5,
               micron_tile_size=__microns_tile_size, crop_emptiness_degree=0.9, label_binning=True,
               subsample_tiles_count=11),
    #
    # val=dict(type=__data_type, path=__psga_dirpath, phase="val", fold=__fold, tiles_intersection=0.0,
    #          micron_tile_size=__microns_tile_size, crop_emptiness_degree=0.95, label_binning=True),

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
MODEL = dict(type="timm.models.senet.seresnext50_32x4d", pretrained=True, num_classes=__classes)

OPTIMIZER = dict(type="torch.optim.SGD", lr=0.01, momentum=0.9, weight_decay=0.0005, dampening=0, nesterov=False)

SCHEDULER = dict(type="torch.optim.lr_scheduler.MultiStepLR", last_epoch=-1, gamma=0.1,
                 milestones=[50, 1200, 5000, 10000])

LOSSES = dict(bce_loss=dict(type="torch.nn.BCEWithLogitsLoss", reduction="mean", pos_weight=None))

METRICS = dict(qwk_metric=dict(type="src.psga.train.evaluation.metric.QuadraticWeightedKappa",
                               labels=None, sample_weight=None),
               accuracy_metric=dict(type="src.psga.train.evaluation.metric.Accuracy", top_k=1))

BATCH_PROCESSOR = dict(val_batch=380) # DP 400 DDP 90 APEX 150


# Hook settings
HOOKS = [
    dict(type="ModifiedProgressBarHook", bar_width=10),
    # dict(type="ModifiedTextLoggerHook"),
    # dict(type="TensorboardLoggerHook", log_dir=f"{WORK_DIR}/tensorboard"),
    # dict(type="ModifiedCheckpointHook", metric_name="val_qwk", num_checkpoints=20, mode="max",
    #      save_optimizer=True, save_scheduler=True),
    dict(type="LRSchedulerHook", name="base", metric_name="", by_epoch=False),

    dict(type="ModelFreezeHook", modules=["layer0", "layer1", "layer2", "layer3", "layer4"],
         train=False, unfreeze_epoch=3),
    dict(type="NormalizationLockHook", train=False, requires_grad=None),

    dict(type="ModifiedPytorchDPHook"),
    dict(type="OptimizerHook", name="base"),

    dict(type="EpochMetricHook", name="train_qwk", runner_metric="qwk_metric"),
    dict(type="EpochMetricHook", name="val_qwk", runner_metric="qwk_metric")
]
