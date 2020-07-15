from typing import (
	Any,
	Dict,
	NoReturn
)

import torch
import torch.nn as nn

from ..contrib.base import BaseBatchProcessor
from ..evaluation.functional import (
    decode_ordinal_logits,
    decode_ordinal_targets
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

        predictions = decode_ordinal_logits(logits)
        dense_targets = decode_ordinal_targets(targets)
        metric = self.estimate("qwk_metric", predictions, dense_targets)

        return dict(
            base_loss=loss,
            values={"base_loss": loss.item(), "batch_qwk": metric.item()},
            num_samples=batch["target"].size(0),
            processed_batch = dict(metric_name="train_qwk", items=batch["item"],
                                   predictions=predictions, targets=dense_targets),
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
            values=dict(),
            num_samples=targets.size(0),
            processed_batch=dict(metric_name="val_qwk", items=batch["item"],
                                 predictions=predictions, targets=decode_ordinal_targets(targets))
        )

    def test_step(self, model: nn.Module, batch: Dict[str, torch.Tensor], **kwargs) -> Dict[str, Any]:
        return self.val_step(model, batch, **kwargs)
