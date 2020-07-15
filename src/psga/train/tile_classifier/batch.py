from typing import (
	Any,
	Dict,
	Tuple,
	Optional,
	NoReturn
)

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..contrib.base import BaseBatchProcessor



class TileClassifierBatchProcessor(BaseBatchProcessor):

    def train_step(self, model: nn.Module, batch: Dict[str, torch.Tensor], **kwargs) -> Dict[str, Any]:



        a = 4

    def val_step(self, model: nn.Module, batch: Dict[str, torch.Tensor], **kwargs) -> Dict[str, Any]:


        a = 4

    def test_step(self, model: nn.Module, batch: Dict[str, torch.Tensor], **kwargs) -> Dict[str, Any]:
        return self.val_step(model, batch, **kwargs)
