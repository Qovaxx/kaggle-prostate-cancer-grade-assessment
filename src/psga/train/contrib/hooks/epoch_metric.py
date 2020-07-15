from typing import (
	ClassVar,
	Dict,
	NoReturn,
	Union,
	List,
	Set
)

import torch
import torch.distributed as dist
from ppln.runner import Runner
from ppln.hooks.registry import HOOKS
from ppln.hooks.base import BaseHook
from ppln.hooks.priority import Priority
from ppln.utils.misc import get_dist_info



@HOOKS.register_module
class EpochMetricHook(BaseHook):
	accepted_keys: ClassVar[Set[str]] = {"metric_name", "items", "predictions", "targets"}

	def __init__(self, metric_name: Union[str, List[str]], epoch_metric: str,
	             pred_dim: int = 0, target_dim: int = 0,
	             outputs_key: str = "processed_batch") -> NoReturn:
		self._metric_name = metric_name
		self._epoch_metric = epoch_metric
		self._pred_dim = pred_dim
		self._target_dim = target_dim
		self._outputs_key = outputs_key
		self._accumulator = list()

	@property
	def priority(self) -> Priority:
		return Priority.NORMAL

	def before_epoch(self, runner: Runner) -> NoReturn:
		self._accumulator.clear()

	def after_iter(self, runner: Runner) -> NoReturn:
		processed_batch: Dict[str, Union[str, torch.Tensor]] = runner.outputs.get(self._outputs_key, None)
		if processed_batch:
			assert self.accepted_keys == set(processed_batch.keys()), \
				f"The keys specified for '{self._outputs_key}' must match with {self.accepted_keys}"
			if self._metric_name == processed_batch["metric_name"]:
				processed_batch = {k: v.cpu().detach() for k, v in processed_batch.items() if k != "metric_name"}
				self._accumulator.append(processed_batch)

	def after_epoch(self, runner: Runner) -> NoReturn:
		if self._accumulator:
			is_distributed = dist.is_initialized()
			items = self._get("items", dim=0, is_distributed=is_distributed)
			predictions = self._get("predictions", dim=self._pred_dim, is_distributed=is_distributed)
			targets = self._get("targets", dim=self._target_dim, is_distributed=is_distributed)

			indices = items.argsort()
			predictions = predictions.index_select(self._pred_dim, indices)
			targets = targets.index_select(self._target_dim, indices)
			metric = runner.batch_processor.estimate(self._epoch_metric, predictions, targets)
			runner.log_buffer.output[self._metric_name] = metric.item()

	def _get(self, key: str, dim: int, is_distributed: bool) -> torch.Tensor:
		tensor = torch.cat([x[key] for x in self._accumulator], dim=dim)
		if is_distributed:
			_, world_size = get_dist_info()
			tensors_gather = [tensor.clone() for _ in range(world_size)]
			dist.all_gather(tensors_gather, tensor)
			tensor =  torch.cat(tensors_gather, dim=dim)
		return tensor
