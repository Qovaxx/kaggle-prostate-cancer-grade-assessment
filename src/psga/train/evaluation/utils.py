import torch


def nan_to_num(input: torch.Tensor, nan_value: float = 0.) -> torch.Tensor:
    if torch.all(torch.isfinite(input)):
        return input
    if len(input.size()) == 0:
        return torch.tensor(nan_value)
    return torch.cat([nan_to_num(l).unsqueeze(0) for l in input], 0)
