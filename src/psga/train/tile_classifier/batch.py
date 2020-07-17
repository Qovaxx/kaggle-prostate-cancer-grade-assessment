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
from ..utils import to_chunks



class TileClassifierBatchProcessor(BaseBatchProcessor):

    def __init__(self, *args, **kwargs) -> NoReturn:
        super().__init__(*args, **kwargs)
        self._val_batch = self._config.BATCH_PROCESSOR.val_batch

    def train_step(self, model: nn.Module, batch: Dict[str, torch.Tensor], **kwargs) -> Dict[str, Any]:
        batch_items = batch["item"]
        batch_images = batch["image"].to(self._experiment.device)
        batch_targets = batch["target"].to(self._experiment.device)

        logits = model(batch_images)
        loss = self.estimate("bce_loss", logits, batch_targets)

        predictions = decode_ordinal_logits(logits)
        targets = decode_ordinal_targets(batch_targets)
        qwk = self.estimate("qwk_metric", predictions, targets)
        acc = self.estimate("acc_metric", logits, targets)

        return dict(
            base_loss=loss,
            values=dict(base_loss=loss.item(), batch_qwk=qwk.item(), batch_acc=acc.item()),
            num_samples=batch_targets.size(0),
            epoch_metric = dict(
                qwk=dict(items=batch_items, inputs=predictions, targets=targets),
                acc=dict(items=batch_items, inputs=logits, targets=targets)
            )
        )

    def val_step(self, model: nn.Module, batch: Dict[str, torch.Tensor], **kwargs) -> Dict[str, Any]:
        batch_items = batch["item"]
        batch_images = batch["image"]
        batch_targets = batch["target"]

        predictions_accum = list()
        logits_accum = list()

        for chunked_images in to_chunks(batch_images, self._val_batch):
            chunked_images = chunked_images.to(self._experiment.device)
            logits = model(chunked_images)
            predictions = decode_ordinal_logits(logits)
            predictions_accum.append(predictions)
            logits_accum.append(logits)

        predictions_accum = torch.cat(predictions_accum)
        logits_accum = torch.cat(logits_accum)

        return dict(
            values=dict(),
            num_samples=batch_targets.size(0),
            epoch_metric=dict(
                qwk=dict(items=batch_items, inputs=predictions_accum, targets=batch_targets),
                acc=dict(items=batch_items, inputs=logits_accum, targets=batch_targets)
            )
        )

    def test_step(self, model: nn.Module, batch: Dict[str, torch.Tensor], **kwargs) -> Dict[str, Any]:
        return self.val_step(model, batch, **kwargs)
