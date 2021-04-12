import torch.nn as nn

from typing import Any
from typing import List
from typing import Optional

from ..types import tensor_dict_type
from ..protocol import TrainerState
from ..protocol import ModelProtocol
from ..constants import INPUT_KEY
from ..constants import PREDICTIONS_KEY
from ..modules.blocks import Mapping


class RNN(ModelProtocol):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        padding_idx: Optional[int] = 0,
        hidden_size: int = 256,
        embedding_dim: int = 256,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        head_activation: str = "ReLU",
        head_batch_norm: bool = True,
        head_dropout: float = 0.5,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx)
        self.gru = nn.GRU(
            embedding_dim,
            hidden_size,
            num_layers,
            bias,
            batch_first,
            dropout,
            bidirectional,
        )
        in_dim = hidden_size
        blocks: List[nn.Module] = []
        for num_unit in [2 * hidden_size] * 2:
            mapping = Mapping(
                in_dim,
                num_unit,
                bias,
                head_activation,
                head_batch_norm,
                head_dropout,
            )
            blocks.append(mapping)
            in_dim = num_unit
        blocks.append(nn.Linear(in_dim, num_classes, bias))
        self.head = nn.Sequential(*blocks)

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: TrainerState,
        **kwargs: Any,
    ) -> tensor_dict_type:
        net = batch[INPUT_KEY]
        net = self.embedding(net)
        num_words = batch["num_words"]
        net, _ = self.gru(net)
        net = net[range(len(net)), num_words - 1]
        net = self.head(net)
        return {PREDICTIONS_KEY: net}


__all__ = ["RNN"]
