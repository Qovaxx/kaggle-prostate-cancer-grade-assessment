from typing import (
	NoReturn,
	Tuple
)

from ppln.runner import Runner
from ppln.hooks.registry import HOOKS
from ppln.hooks.priority import Priority
from ppln.hooks.resume import ResumeHook

from ..utils.checkpoint import load_checkpoint


@HOOKS.register_module
class ModifiedResumeHook(ResumeHook):

	def __init__(self, checkpoint: str, resume_optimizer: bool = True, resume_scheduler: bool = True,
	             resume_iter: bool = True, strict: bool = False, map_location: str = "cpu",
	             ignore_loaded_keys: Tuple[str] = ()) -> NoReturn:
		super().__init__(checkpoint, resume_optimizer, resume_scheduler, resume_iter,
		                 strict, map_location)
		self._ignore_loaded_keys = ignore_loaded_keys

	@property
	def priority(self) -> Priority:
		return Priority.HIGHEST # Must be bigger than FreezeHook, LockBatchNormHook and DDP Hooks

	def before_run(self, runner: Runner) -> NoReturn:
		runner.logger.info(f"Resume from {self.checkpoint}")
		checkpoint = load_checkpoint(
			runner.model,
			self.checkpoint,
			map_location=self.map_location,
			strict=self.strict,
			optimizer=runner.optimizers if self.resume_optimizer else None,
			scheduler=runner.schedulers if self.resume_scheduler else None,
			ignore_loaded_keys=self._ignore_loaded_keys
		)

		if self.resume_iter:
			runner.epoch = checkpoint["meta"]["epoch"]
			runner.iter = checkpoint["meta"]["iter"]
			runner.logger.info(f"resumed epoch: {runner.epoch}, iter: {runner.iter}")
