from collections import OrderedDict
from typing import (
    Tuple,
    Optional
)

import torch
from ppln.utils.checkpoint import (
    load_optim_state_dict,
    load_state_dict
)
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer



def load_checkpoint(model: torch.nn.Module, filename: str, map_location: Optional[str] = None,
                    strict: bool = False, optimizer: Optional[Optimizer] = None,
                    scheduler: Optional[_LRScheduler] = None, ignore_loaded_keys: Tuple[str] = ()):
    """Load checkpoint from a file or URI."""
    checkpoint = torch.load(filename, map_location=map_location)

    # Get state_dict from checkpoint
    if isinstance(checkpoint, OrderedDict):
        state_dict = checkpoint
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        raise RuntimeError("No state_dict found in checkpoint file {}".format(filename))

    # Strip prefix of state_dict
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k[7:]: v for k, v in checkpoint["state_dict"].items()}

    # Load state_dict
    state_dict = OrderedDict({k:v for k, v in state_dict.items() if not any([ik in k for ik in ignore_loaded_keys])})
    if hasattr(model, "module"):
        load_state_dict(model.module, state_dict, strict)
    else:
        load_state_dict(model, state_dict, strict)

    if "optimizer" in checkpoint and optimizer is not None:
        load_optim_state_dict(optimizer, checkpoint["optimizer"], Optimizer)
    if "scheduler" in checkpoint and scheduler is not None:
        load_optim_state_dict(scheduler, checkpoint["scheduler"], _LRScheduler)

    return checkpoint
