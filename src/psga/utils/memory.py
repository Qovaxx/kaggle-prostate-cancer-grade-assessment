import gc

import torch
import numpy as np


def reduce_numpy_memory(image: np.ndarray) -> np.ndarray:
    image_gpu = torch.from_numpy(image).cuda()
    del image
    gc.collect()
    image = image_gpu.cpu().numpy()
    del image_gpu
    torch.cuda.empty_cache()
    return image
