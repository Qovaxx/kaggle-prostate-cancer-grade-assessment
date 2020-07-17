from collections import defaultdict
from typing import (
    ClassVar,
    Dict,
    List,
    NoReturn,
    Set
)

import torch
import torch.distributed as dist
from ppln.runner import Runner
from ppln.hooks.registry import HOOKS
from ppln.hooks.base import BaseHook
from ppln.hooks.priority import Priority
from ppln.utils.misc import get_dist_info

from ..utils.dist import all_gather_cpu


@HOOKS.register_module
class EpochMetricHook(BaseHook):

    accepted_keys: ClassVar[Set[str]] = {"items", "inputs", "targets"}

    def __init__(self, handle: Dict[str, str],
                 inputs_dim: int = 0, targets_dim: int = 0,
                 outputs_key: str = "epoch_metric") -> NoReturn:
        self._handle = handle
        self._inputs_dim = inputs_dim
        self._targets_dim = targets_dim
        self._outputs_key = outputs_key
        self._accumulator = defaultdict(list)
        self._is_distributed = dist.is_initialized()

    @property
    def priority(self) -> Priority:
        return Priority.NORMAL

    def before_epoch(self, runner: Runner) -> NoReturn:
        self._accumulator.clear()

    def after_iter(self, runner: Runner) -> NoReturn:
        predictions: Dict[str, Dict[str, torch.Tensor]] = runner.outputs.get(self._outputs_key, None)
        if predictions and all([self.accepted_keys == set(v.keys()) for k, v in predictions.items()]):

            for metric in predictions.keys():
                batch = predictions[metric]
                self._accumulator[metric].append({k: v.cpu().detach() for k, v in batch.items()})

    def after_epoch(self, runner: Runner) -> NoReturn:
        for metric_name in self._handle.keys():
            accumulated = self._accumulator.get(metric_name, None)

            if accumulated:
                items = self._flat(accumulated, key="items", dim=0)
                inputs = self._flat(accumulated, key="inputs", dim=self._inputs_dim)
                targets = self._flat(accumulated, key="targets", dim=self._targets_dim)

                indices = items.argsort()
                inputs = inputs.index_select(self._inputs_dim, indices)
                targets = targets.index_select(self._targets_dim, indices)

                rank, _ = get_dist_info()
                if rank == 0:
                    func_name = self._handle[metric_name]
                    metric = runner.batch_processor.estimate(func_name, inputs, targets)
                    runner.log_buffer.output[metric_name] = metric.item()

    def _flat(self, data: List[Dict[str, torch.Tensor]], key: str, dim: int) -> torch.Tensor:
        tensor = torch.cat([batch[key] for batch in data], dim=dim)
        if self._is_distributed:
            gathered_tensors = all_gather_cpu(tensor)
            tensor = torch.cat(gathered_tensors, dim=dim)

        return tensor
