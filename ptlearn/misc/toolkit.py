import torch
import datetime

import numpy as np
import torch.nn as nn

from typing import Any
from typing import Dict
from typing import Type
from typing import Callable
from typing import Optional
from typing import ContextManager

from ..types import tensor_dict_type
from ..constants import WARNING_PREFIX


def is_int(arr: np.ndarray) -> bool:
    return np.issubdtype(arr.dtype, np.integer)


def is_float(arr: np.ndarray) -> bool:
    return np.issubdtype(arr.dtype, np.floating)


def to_standard(arr: np.ndarray) -> np.ndarray:
    if is_int(arr):
        arr = arr.astype(np.int64)
    elif is_float(arr):
        arr = arr.astype(np.float32)
    return arr


def to_torch(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(to_standard(arr))


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def to_device(batch: tensor_dict_type, device: torch.device) -> tensor_dict_type:
    return {
        k: v if not isinstance(v, torch.Tensor) else v.to(device)
        for k, v in batch.items()
    }


def timestamp(simplify: bool = False, ensure_different: bool = False) -> str:
    """
    Return current timestamp.

    Parameters
    ----------
    simplify : bool. If True, format will be simplified to 'year-month-day'.
    ensure_different : bool. If True, format will include millisecond.

    Returns
    -------
    timestamp : str

    """

    now = datetime.datetime.now()
    if simplify:
        return now.strftime("%Y-%m-%d")
    if ensure_different:
        return now.strftime("%Y-%m-%d_%H-%M-%S-%f")
    return now.strftime("%Y-%m-%d_%H-%M-%S")


def shallow_copy_dict(d: dict) -> dict:
    d = d.copy()
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = shallow_copy_dict(v)
    return d


def fix_float_to_length(num: float, length: int) -> str:
    """ Change a float number to string format with fixed length. """

    str_num = f"{num:f}"
    if str_num == "nan":
        return f"{str_num:^{length}s}"
    length = max(length, str_num.find("."))
    return str_num[:length].ljust(length, "0")


def register_core(
    name: str,
    global_dict: Dict[str, type],
    *,
    before_register: Optional[Callable] = None,
    after_register: Optional[Callable] = None,
) -> Callable[[Type], Type]:
    def _register(cls: Type) -> Type:
        if before_register is not None:
            before_register(cls)
        registered = global_dict.get(name)
        if registered is not None:
            print(
                f"{WARNING_PREFIX}'{name}' has already registered "
                f"in the given global dict ({global_dict})"
            )
            return cls
        global_dict[name] = cls
        if after_register is not None:
            after_register(cls)
        return cls

    return _register


class context_error_handler:
    """ Util class which provides exception handling when using context manager. """

    def _normal_exit(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Any:
        pass

    def _exception_exit(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Any:
        pass

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Any:
        if not exc_type:
            self._normal_exit(exc_type, exc_val, exc_tb)
        else:
            self._exception_exit(exc_type, exc_val, exc_tb)


class mode_context(context_error_handler):
    """
    Help entering specific mode and recovering previous mode

    This is a context controller for entering specific mode at the beginning
    and back to previous mode at the end.

    Parameters
    ----------
    module : nn.Module, arbitrary PyTorch module.

    Examples
    --------
    >>> module = nn.Module()
    >>> with mode_context(module):
    >>>     pass  # do something

    """

    def __init__(
        self,
        module: nn.Module,
        *,
        to_train: Optional[bool],
        use_grad: Optional[bool],
    ):
        self._to_train = to_train
        self._module, self._training = module, module.training
        self._cache = {p: p.requires_grad for p in module.parameters()}
        if use_grad is not None:
            for p in module.parameters():
                p.requires_grad_(use_grad)
        if use_grad is None:
            self._grad_context: Optional[ContextManager] = None
        else:
            self._grad_context = torch.enable_grad() if use_grad else torch.no_grad()

    def __enter__(self) -> None:
        if self._to_train is not None:
            self._module.train(mode=self._to_train)
        if self._grad_context is not None:
            self._grad_context.__enter__()

    def _normal_exit(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._to_train is not None:
            self._module.train(mode=self._training)
        if self._grad_context is not None:
            self._grad_context.__exit__(exc_type, exc_val, exc_tb)
        for p, v in self._cache.items():
            p.requires_grad_(v)


class train_context(mode_context):
    """
    Useful when we need to get gradients with our PyTorch model during evaluating.
    """

    def __init__(self, module: nn.Module, *, use_grad: bool = True):
        super().__init__(module, to_train=True, use_grad=use_grad)


class eval_context(mode_context):
    """
    Useful when we need to predict something with our PyTorch model during training.
    """

    def __init__(self, module: nn.Module, *, use_grad: bool = False):
        super().__init__(module, to_train=False, use_grad=use_grad)
