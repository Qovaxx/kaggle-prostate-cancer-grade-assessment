import sys
from typing import NoReturn

from colorama import (
    Fore,
    Style
)
from ppln.runner import Runner
from ppln.hooks.registry import HOOKS
from ppln.hooks.logger.progress_bar import ProgressBarLoggerHook
from ppln.hooks.logger.utils import get_lr
from ppln.utils.misc import master_only

from ..utils.progress_bar import ModifiedProgressBar



@HOOKS.register_module
class ModifiedProgressBarHook(ProgressBarLoggerHook):

    def before_epoch(self, runner: Runner) -> NoReturn:
        self.bar = ModifiedProgressBar(task_num=len(runner.data_loader), bar_width=self.bar_width)

    @master_only
    def after_epoch(self, runner: Runner):
        self.log(runner, update_completed=False)
        sys.stdout.write(f"\n")

    def after_iter(self, runner):
        self.log(runner, update_completed=True)

    @master_only
    def log(self, runner: Runner, **kwargs):
        epoch_color = Fore.YELLOW
        mode_color = (Fore.RED, Fore.BLUE)[runner.train_mode]
        text_color = (Fore.CYAN, Fore.GREEN)[runner.train_mode]
        epoch_text = f"{epoch_color}epoch:{Style.RESET_ALL} {runner.epoch + 1:<4}"
        log_items = [(" " * 11, epoch_text)[runner.train_mode], f"{mode_color}{runner.mode:<5}{Style.RESET_ALL}"]
        log_items.append(f"{text_color}iter:{Style.RESET_ALL} {runner.iter + 1}")

        for name, lrs in get_lr(runner.optimizers).items():
            log_items.append(f"{text_color}{name}_lr:{Style.RESET_ALL} {', '.join([f'{lr:.3e}' for lr in lrs])}")

        for name, value in runner.log_buffer.output.items():
            if isinstance(value, float):
                    value = f"{value:.2f}" if name in ["data_time", "time"] else f"{value:.4f}"
            log_items.append(f'{text_color}{name}:{Style.RESET_ALL} {value}')
        self.bar.update(f"{' | '.join(log_items)}", kwargs["update_completed"])
