import torch
from ppln.utils.misc import object_from_dict
from ppln.utils.config import ConfigDict


def make_model(config: ConfigDict, device: torch.device = torch.device("cpu")) -> torch.nn.Module:
    model = object_from_dict(config)
    return model.to(device)
