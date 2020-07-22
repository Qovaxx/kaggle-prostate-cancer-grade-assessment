from logging import Logger
from functools import partial
from typing import (
	NoReturn,
	Optional,
	List,
)

from ppln.runner import Runner
from ppln.hooks.registry import HOOKS
from ppln.hooks.base import BaseHook
from ppln.hooks.priority import Priority
from ppln.utils.misc import master_only

from ..utils.freeze import (
	freeze_modules,
	lock_norm_modules
)


@HOOKS.register_module
class ModelFreezeHook(BaseHook):

	def __init__(self, modules: List[str], train: bool = False, unfreeze_epoch: Optional[int] = None) -> NoReturn:
		self._func = partial(freeze_modules, root_modules=modules)
		self._train = train
		self._unfreeze_epoch = float("inf") if unfreeze_epoch is None else unfreeze_epoch

	@property
	def priority(self) -> Priority:
		return Priority.VERY_HIGH # Must be higher than DDPHook and NormalizationLockHook

	@master_only
	def _logit(self, message: str, logger: Logger):
		logger.info(message)

	def before_run(self, runner: Runner) -> NoReturn:
		self._func(model=runner.model, requires_grad=False, train=self._train)
		self._logit("Layers are frozen", runner.logger)

	def before_train_epoch(self, runner: Runner) -> NoReturn:
		unfreeze_epoch = self._unfreeze_epoch - 1

		if runner.epoch < unfreeze_epoch:
			self._func(model=runner.model, requires_grad=False, train=self._train)
			self._logit("Layers are re-frozen", runner.logger)

		elif runner.epoch == unfreeze_epoch:
			self._func(model=runner.model, requires_grad=True, train=True)
			self._logit("Layers are de-frozen", runner.logger)


@HOOKS.register_module
class NormalizationLockHook(BaseHook):

	def __init__(self, train: bool = False, requires_grad: Optional[bool] = None) -> NoReturn:
		self._func = partial(lock_norm_modules, train=train, requires_grad=requires_grad)

	@property
	def priority(self) -> Priority:
		return Priority.HIGH # Must be higher than ddp wrapper

	def before_run(self, runner: Runner) -> NoReturn:
		self._func(model=runner.model)

	def before_train_epoch(self, runner: Runner) -> NoReturn:
		self._func(model=runner.model)