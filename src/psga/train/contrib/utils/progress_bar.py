import sys
from typing import NoReturn

from ppln.utils.progress_bar import ProgressBar


class ModifiedProgressBar(ProgressBar):

	def update(self, text: str = "", update_completed: bool = True) -> NoReturn:
		if update_completed:
			self.completed += 1
		elapsed = self.timer.since_start()
		fps = self.completed / elapsed

		if self.task_num > 0:
			percentage = self.completed / float(self.task_num)
			eta = int(elapsed * (1 - percentage) / percentage + 0.5)
			mark_width = int(self.bar_width * percentage)
			bar_chars = ">" * mark_width + " " * (self.bar_width - mark_width)
			sys.stdout.write(
				f"\r{text} "
				f"[{bar_chars}] "
				f"{self.completed}/{self.task_num}, "
				f"{fps:.1f} task/s, "
				f"ET: {int(elapsed + 0.5)}s, "
				f"ETA: {eta:5}s"
			)
		else:
			sys.stdout.write(f"{text} completed: {self.completed}, ET: {int(elapsed + 0.5)}s, {fps:.1f} tasks/s")
		sys.stdout.flush()
