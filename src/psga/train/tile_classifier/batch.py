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
        acc = self.estimate("accuracy_metric", logits, dense_targets)
        qwk = self.estimate("qwk_metric", predictions, dense_targets)

        return dict(
            base_loss=loss,
            values={"base_loss": loss.item(), "qwk": qwk.item(), "acc": acc.item()},
            num_samples=batch["target"].size(0),
            processed_batch = dict(metric_name="train_qwk", items=batch["item"],
                                   predictions=predictions, targets=dense_targets),
        )

    def val_step(self, model: nn.Module, batch: Dict[str, torch.Tensor], **kwargs) -> Dict[str, Any]:
        images = batch["image"]
        targets = batch["target"]
        images_chunks = split_to_chunks(images, chunk_size=self._val_batch)
        targets_chunks = split_to_chunks(targets, chunk_size=self._val_batch)

        predictions = list()
        acc = list()

        for chunk_images, chunk_targets in zip(images_chunks, targets_chunks):
            chunk_images = chunk_images.to(self._experiment.device)
            chunk_targets = chunk_targets.to(self._experiment.device)

            chunk_logits = model(chunk_images)
            predictions.append(decode_ordinal_logits(chunk_logits))
            acc.append(self.estimate("accuracy_metric", chunk_logits, decode_ordinal_targets(chunk_targets)))

        acc = torch.cat(acc).mean()
        predictions = torch.cat(predictions)

        return dict(
            values={"acc": acc.item()},
            num_samples=targets.size(0),
            processed_batch=dict(metric_name="val_qwk", items=batch["item"],
                                 predictions=predictions, targets=decode_ordinal_targets(targets))
        )

    def test_step(self, model: nn.Module, batch: Dict[str, torch.Tensor], **kwargs) -> Dict[str, Any]:
        return self.val_step(model, batch, **kwargs)
