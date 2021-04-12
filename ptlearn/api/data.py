import numpy as np

from typing import Any
from typing import Callable
from typing import Optional
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from ..types import tensor_dict_type
from ..protocol import DataProtocol
from ..protocol import DataLoaderProtocol
from ..constants import INPUT_KEY
from ..constants import LABEL_KEY
from ..misc.toolkit import to_torch


@DataProtocol.register("ml")
class MLData(DataProtocol):
    def __init__(self, x: np.ndarray, y: Optional[np.ndarray]):
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return len(self.x)


@DataLoaderProtocol.register("ml")
class MLLoader(DataLoaderProtocol):
    data: MLData
    cursor: int
    indices: np.ndarray

    def __init__(
        self,
        data: MLData,
        shuffle: bool,
        *,
        batch_size: int = 128,
    ):
        self.data = data
        self.shuffle = shuffle
        self.batch_size = batch_size

    def __iter__(self) -> "MLLoader":
        self.cursor = 0
        self.indices = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self

    def __next__(self) -> tensor_dict_type:
        start = self.cursor
        if start >= len(self.data):
            raise StopIteration
        self.cursor += self.batch_size
        indices = self.indices[start : self.cursor]
        return {
            INPUT_KEY: to_torch(self.data.x[indices]),
            LABEL_KEY: None if self.data.y is None else to_torch(self.data.y[indices]),
        }


@DataProtocol.register("dl")
class DLData(DataProtocol):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore

    def __getitem__(self, item: Any) -> Any:
        return self.dataset[item]


@DataLoaderProtocol.register("dl")
class DLLoader(DataLoaderProtocol):
    data: DLData

    def __init__(
        self,
        loader: DataLoader,
        batch_callback: Callable[[Any], tensor_dict_type],
    ):
        self.loader = loader
        self.data = loader.dataset  # type: ignore
        self.batch_size = loader.batch_size  # type: ignore
        self.batch_callback = batch_callback
        self._iterator: Optional[Any] = None

    def __iter__(self) -> "DLLoader":
        self._iterator = self.loader.__iter__()
        return self

    def __next__(self) -> tensor_dict_type:
        return self.batch_callback(self._iterator.__next__())  # type: ignore


__all__ = [
    "MLData",
    "MLLoader",
    "DLData",
    "DLLoader",
]
