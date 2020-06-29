import os
from operator import itemgetter
from itertools import chain
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


def mimic_kaggle_kernel_specs(cpu: bool = True, ram: bool = True) -> NoReturn:
    system = System()
    if cpu:
        system.limit_cpu(cpus=[0, 1], logical=True)
    if ram:
        system.limit_ram(gigabytes=23.5)


class System(object):

    def __init__(self) -> NoReturn:
        self._cpu_physical_cores_count = cpu_count(logical=False)
        self._cpu_logical_cores_count = cpu_count(logical=True)
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

    def limit_ram(self, gigabytes: float) -> NoReturn:
        assert gigabytes <= self._memory_total_ram
        _, hard = getrlimit(RLIMIT_AS)
        setrlimit(RLIMIT_AS, (gigabytes * 1024 ** 3, hard))
