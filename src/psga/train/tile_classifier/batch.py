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
from ..utils import split_to_chunks



class TileClassifierBatchProcessor(BaseBatchProcessor):

    def __init__(self, *args, **kwargs) -> NoReturn:
        super().__init__(*args, **kwargs)
        self._val_batch = self._config.BATCH_PROCESSOR.val_batch

    def train_step(self, model: nn.Module, batch: Dict[str, torch.Tensor], **kwargs) -> Dict[str, Any]:
        images = batch["image"].to(self._experiment.device)
        targets = batch["target"].to(self._experiment.device)

        logits = model(images)
        loss = self.estimate("bce_loss", logits, targets)
        metric = self.estimate("qwk_metric", decode_ordinal_logits(logits), decode_ordinal_labels(targets))

        return dict(
            base_loss=loss,
            values={"base_loss": loss.item(), "batch_qwk": metric.item()},
            num_samples=batch["target"].size(0)
        )

    def val_step(self, model: nn.Module, batch: Dict[str, torch.Tensor], **kwargs) -> Dict[str, Any]:
        images = batch["image"]
        targets = batch["target"]
        predictions = list()

        for chunk in split_to_chunks(images, chunk_size=self._val_batch):
            chunk = chunk.to(self._experiment.device)
            logits = model(chunk)
            predictions.append(decode_ordinal_logits(logits))

        predictions = torch.cat(predictions)

        return dict(
            processed_batch=dict(metric_name="qwk", items=batch["item"])
        )




        a = 4

    def test_step(self, model: nn.Module, batch: Dict[str, torch.Tensor], **kwargs) -> Dict[str, Any]:
        return self.val_step(model, batch, **kwargs)
