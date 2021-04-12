import torch

import torch.nn as nn

from typing import Any
from typing import List
from typing import Optional

from ..types import tensor_dict_type
from ..protocol import TrainerState
from ..protocol import ModelProtocol
from ..constants import INPUT_KEY
from ..constants import PREDICTIONS_KEY
from ..modules.blocks import BN
from ..modules.blocks import Mapping


class FCNN(ModelProtocol):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_units: List[int],
        *,
        bias: bool = True,
        activation: str = "ReLU",
        batch_norm: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        blocks: List[nn.Module] = []
        for num_unit in num_units:
            mapping = Mapping(in_dim, num_unit, bias, activation, batch_norm, dropout)
            blocks.append(mapping)
            in_dim = num_unit
        blocks.append(nn.Linear(in_dim, out_dim, bias))
        self.net = nn.Sequential(*blocks)

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: TrainerState,
        **kwargs: Any,
    ) -> tensor_dict_type:
        return {PREDICTIONS_KEY: self.net(batch[INPUT_KEY])}


__all__ = ["FCNN"]
