import os
import sys
from operator import itemgetter
from itertools import chain
from subprocess import check_output
from typing import (
    NoReturn,
    List,
    Union
)
from psutil import (
    Process,
    cpu_count,
    virtual_memory
)
from resource import (
    RLIMIT_AS,
    getrlimit,
    setrlimit,
)

import numpy as np

DL_FRAMEWORKS = ("torch", "tensorflow", "maxnet", "caffe2", "caffe", "lasagne", "chainer")


def mimic_kaggle_kernel_specs() -> NoReturn:
    system = System()
    system.limit_cpu(cpus=[0, 1], logical=True)
    system.limit_gpu(gpus=[0])
    system.limit_ram(gigabytes=13)


class System(object):

    def __init__(self) -> NoReturn:
        self._cpu_physical_cores_count = cpu_count(logical=False)
        self._cpu_logical_cores_count = cpu_count(logical=True)
        self._gpu_devices_count = str(check_output(["nvidia-smi", "-L"])).count("UUID")
        self._memory_total_ram = virtual_memory().total / 1024 ** 3

    def _force_list(self, choice: Union[int, List[int]]) -> List[int]:
        if isinstance(choice, int):
            choice = [choice]
        return choice

    def limit_cpu(self, cpus: Union[int, List[int]], logical: bool = True) -> NoReturn:
        count = len(cpus)
        if logical:
            assert count <= self._cpu_logical_cores_count
        else:
            assert count <= self._cpu_physical_cores_count
            logical_cores = list(range(self._cpu_logical_cores_count))
            physical_cores = np.array_split(logical_cores, indices_or_sections=self._cpu_physical_cores_count)
            cpus = list(chain(*itemgetter(*cpus)(physical_cores)))

        cpus = self._force_list(cpus)
        process = Process(pid=os.getpid())
        process.cpu_affinity(cpus=cpus)

    def limit_gpu(self, gpus: Union[int, List[int]]) -> NoReturn:
        imported_dl_frameworks = set(DL_FRAMEWORKS).intersection(set(sys.modules))
        assert len(imported_dl_frameworks) == 0
        assert len(gpus) <= self._gpu_devices_count
        gpus = map(str, self._force_list(gpus))
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)

    def limit_ram(self, gigabytes: float) -> NoReturn:
        assert gigabytes <= self._memory_total_ram
        _, hard = getrlimit(RLIMIT_AS)
        setrlimit(RLIMIT_AS, (gigabytes * 1024 ** 3, hard))
