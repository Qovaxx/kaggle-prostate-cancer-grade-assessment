from typing import NoReturn

import torch
from ppln.hooks.registry import HOOKS
from ppln.hooks.priority import Priority
from ppln.factory import make_pytorch_ddp
from ppln.hooks.dist import BaseClosureHook
from torch.nn.parallel import DataParallel

from ..utils.data_parallel import BalancedDataParallel


@HOOKS.register_module
class ModifiedBaseDistClosureHook(BaseClosureHook):

	@property
	def priority(self) -> Priority:
		return Priority.NORMAL

	def before_run(self, runner) -> NoReturn:
		model_device = next(runner.model.parameters()).device
		if model_device != torch.device("cpu"):
			runner.model = self.func(runner.model)


@HOOKS.register_module
class ModifiedPytorchDPHook(ModifiedBaseDistClosureHook):

	def __init__(self, **kwargs) -> NoReturn:
		super().__init__(DataParallel, **kwargs)


@HOOKS.register_module
class ModifiedPytorchBDPHook(ModifiedBaseDistClosureHook):

	def __init__(self, **kwargs) -> NoReturn:
		super().__init__(BalancedDataParallel, **kwargs)


@HOOKS.register_module
class ModifiedPytorchDDPHook(ModifiedBaseDistClosureHook):
	def __init__(self, **kwargs):
		super().__init__(make_pytorch_ddp, **kwargs)
