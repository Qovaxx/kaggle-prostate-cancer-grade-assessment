import datetime
from typing import NoReturn

from ppln.runner import Runner
from ppln.hooks.registry import HOOKS
from ppln.hooks.logger.text import TextLoggerHook


@HOOKS.register_module
class ModifiedTextLoggerHook(TextLoggerHook):

    def _log_info(self, log_dict, runner: Runner) -> NoReturn:
        if runner.mode == "train":
            iter_string = f"iter: {runner.iter + 1}, "
            lr_str = "".join(
                [f"{name}_lr: {', '.join([f'{lr:.3e}' for lr in lrs])}, " for name, lrs in log_dict["lr"].items()])
            log_str =\
                f"Epoch [{log_dict['epoch']}][{log_dict['iter']}/{len(runner.data_loader)}]\t{iter_string}{lr_str}"

            if "time" in log_dict.keys():
                self.time_sec_tot += (log_dict["time"] * len(runner.data_loader))
                time_sec_avg = self.time_sec_tot / (runner.iter - self.start_iter + 1)
                eta_sec = time_sec_avg * (runner.max_iters - runner.iter - 1)
                eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
                log_str += f"eta: {eta_str}, "
                log_str += f"time: {log_dict['time']:.2f}, data_time: {log_dict['data_time']:.2f}, "
        else:
            log_str = f"Epoch({log_dict['mode']}) [{log_dict['epoch']}][{log_dict['iter']}]\t"

        log_items = list()
        for name, value in log_dict.items():
            if name in ["Epoch", "mode", "iter", "lr", "time", "data_time", "epoch"]:
                continue
            value = f"{value:.4f}" if isinstance(value, float) else value
            log_items.append(f"{name}: {value}")
        log_str += ", ".join(log_items)
        runner.logger.info(log_str)
