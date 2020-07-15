from typing import (
	Any,
	Dict,
	Tuple,
	Optional,
	NoReturn
)

import torch
import torch.nn as nn

from ..contrib.base import BaseBatchProcessor
from ..evaluation.functional import (
    decode_ordinal_logits,
    decode_ordinal_labels
)



class TileClassifierBatchProcessor(BaseBatchProcessor):

    def __init__(self, *args, **kwargs) -> NoReturn:
        super().__init__(*args, **kwargs)
        self._loss = self._experiment.losses["bce"]
        self._metric = self._experiment.metrics["qwk"]

    def train_step(self, model: nn.Module, batch: Dict[str, torch.Tensor], **kwargs) -> Dict[str, Any]:
        images = batch["image"].to(self._experiment.device)
        targets = batch["target"].to(self._experiment.device)

        logits = model(images)
        loss = self._loss(logits, targets)
        metric = self._metric(decode_ordinal_logits(logits), decode_ordinal_labels(targets))

        return dict(
            base_loss=loss,
            values={"base_loss": loss.item(), "batch_qwk": metric.item()},
            num_samples=batch["target"].size(0)
        )

    def val_step(self, model: nn.Module, batch: Dict[str, torch.Tensor], **kwargs) -> Dict[str, Any]:


        a = 4

    def test_step(self, model: nn.Module, batch: Dict[str, torch.Tensor], **kwargs) -> Dict[str, Any]:
        return self.val_step(model, batch, **kwargs)
