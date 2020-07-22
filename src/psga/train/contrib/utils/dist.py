import os
import tempfile
import os.path as osp

import torch
import torch.distributed as dist
from ppln.utils.misc import get_dist_info
from ppln.fileio import io


def all_gather_cpu(data, tmpdir=None):
    rank, world_size = get_dist_info()

    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN,), 32, dtype=torch.uint8, device="cuda")
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(bytearray(tmpdir.encode()), dtype=torch.uint8, device="cuda")
            dir_tensor[: len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        os.makedirs(tmpdir, exist_ok=True)
    # dump the part result to the dir
    io.dump(data, osp.join(tmpdir, f"part_{rank}.pkl"))
    dist.barrier()
    # collect all parts
    data_list = []
    for i in range(world_size):
        data = osp.join(tmpdir, f"part_{i}.pkl")
        data_list.append(io.load(data))

    return data_list
