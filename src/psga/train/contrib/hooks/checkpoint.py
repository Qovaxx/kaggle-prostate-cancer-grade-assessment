from ppln.runner import Runner
from ppln.hooks.registry import HOOKS
from ppln.hooks.checkpoint import CheckpointHook
from ppln.utils.misc import master_only


@HOOKS.register_module
class ModifiedCheckpointHook(CheckpointHook):

    @master_only
    def after_val_epoch(self, runner: Runner):
        if self.metric_name in runner.log_buffer.output.keys():
            metric = runner.log_buffer.output[self.metric_name]

            if self.mode == "min":
                metric *= -1

            if self._is_update(metric):
                self._checkpoints.put((metric, self.current_filepath(runner)))
                self._save_checkpoint(runner)
            if self._best_metric < metric:
                self._best_metric = metric
                self._save_link(runner)
                runner.logger.info(
                    f"Best checkpoint was changed: {self.current_filename(runner)} with {self._best_metric}"
                )
