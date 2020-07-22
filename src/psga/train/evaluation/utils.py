import torch


def nan_to_num(inputs: torch.Tensor, nan_value: float = 0.) -> torch.Tensor:
    if torch.all(torch.isfinite(inputs)):
        return inputs
    if len(inputs.size()) == 0:
        return torch.tensor(nan_value)
    return torch.cat([nan_to_num(l).unsqueeze(0) for l in inputs], 0)
