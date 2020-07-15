from typing import List

import torch


def split_to_chunks(batch: torch.Tensor, chunk_size: int, batch_safe: bool = True) -> List[torch.Tensor]:
    chunks = list(torch.split(batch, chunk_size, dim=0))
    if batch_safe:
        gpu_count = torch.cuda.device_count()
        if chunks[-1].size(0) < gpu_count:
            chunks = chunks[:-2] + [torch.cat([chunks[-2], chunks[-1]], dim=0)]
    return chunks
