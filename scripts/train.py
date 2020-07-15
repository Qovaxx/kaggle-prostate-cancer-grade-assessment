import warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn

import argparse
from typing import NoReturn

import torch.distributed as dist
from ppln.utils.config import Config
from ppln.utils.misc import init_dist
from ppln.runner import Runner

from src.psga.train.tile_classifier.builder import TileClassifierDDPBuilder, TileClassifierDPBuilder
from src.psga.train.tile_classifier.batch import TileClassifierBatchProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Process main train arguments")
    parser.add_argument("--config_path", type=str,
                        help="Path to the config file")
    parser.add_argument("--is_distributed", default=False, action="store_true",
                        help="Enable distributed mode")
    parser.add_argument("--local_rank", type=int, default=0,
                        help="The rank of the node for distributed training")
    return parser.parse_args()


def train_tile_classifier(config: Config) -> NoReturn:

    if dist.is_initialized():
        builder = TileClassifierDDPBuilder(config)
    else:
        builder = TileClassifierDPBuilder(config)

    # data_loaders = {x: builder.data_loader(x) for x in config.DATA.keys()}
    batch_processor = TileClassifierBatchProcessor(builder)

    optimizers = builder.optimizers

    a = 4

    # runner = Runner(
    # 	model=builder.model,
    # 	optimizers=builder.optimizers,
    # 	schedulers=builder.schedulers,
    # 	hooks=builder.hooks,
    # 	work_dir=builder.config.WORK_DIR,
    # 	batch_processor=batch_processor,
    # )
    # runner.run(
    # 	data_loaders=data_loaders,
    # 	max_epochs=builder.config.MAX_EPOCHS
    # )


if __name__ == "__main__":
    args = parse_args()
    config = Config.fromfile(args.config_path)

    if args.is_distributed:
        init_dist(**config.DIST_PARAMS)

    train_func = locals()[config.TRAIN_FUNC]
    train_func(config)
