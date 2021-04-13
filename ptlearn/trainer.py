import os
import copy
import json
import torch
import shutil

import numpy as np

from typing import *
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from .constants import *
from .types import tensor_dict_type
from .protocol import StepOutputs
from .protocol import LossProtocol
from .protocol import TrainerState
from .protocol import ModelProtocol
from .protocol import MetricsOutputs
from .protocol import MonitorResults
from .protocol import TrainerMonitor
from .protocol import MetricsProtocol
from .protocol import InferenceOutputs
from .protocol import InferenceProtocol
from .protocol import DataLoaderProtocol
from .misc.toolkit import to_device
from .misc.toolkit import timestamp
from .misc.toolkit import fix_float_to_length
from .modules.optimizers import optimizer_dict
from .modules.schedulers import scheduler_dict


class TrainerCallback:
    is_rank_0: bool = True

    def log_lr(self, key: str, lr: float, step: int) -> None:
        pass

    def log_metrics(self, metric_outputs: MetricsOutputs) -> None:
        pass

    def log_metrics_msg(
        self,
        state: TrainerState,
        metric_outputs: MetricsOutputs,
        metric_log_path: str,
    ) -> None:
        if not self.is_rank_0:
            return None
        final_score = metric_outputs.final_score
        metric_values = metric_outputs.metric_values
        core = " | ".join(
            [
                f"{k} : {fix_float_to_length(metric_values[k], 8)}"
                for k in sorted(metric_values)
            ]
        )
        msg = (
            f"| epoch {state.epoch:^4d} - "
            f"step {state.step:^6d} | {core} | "
            f"score : {fix_float_to_length(final_score, 8)} |"
        )
        print(msg)
        with open(metric_log_path, "a") as f:
            f.write(f"{msg}\n")

    def log_artifacts(self, trainer: "Trainer") -> None:
        pass

    def after_step(self, step_outputs: StepOutputs) -> None:
        pass

    def after_monitor(self, monitor_results: MonitorResults) -> None:
        pass


class Trainer:
    loss: LossProtocol
    model: ModelProtocol
    state: TrainerState
    device: torch.device
    train_loader: DataLoaderProtocol
    train_loader_copy: DataLoaderProtocol
    valid_loader: Optional[DataLoaderProtocol]
    inference: InferenceProtocol

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        state_config: Optional[Dict[str, Any]] = None,
        *,
        num_epoch: int = 10,
        valid_portion: float = 1.0,
        amp: bool = False,
        clip_norm: float = 0.0,
        monitor: Optional[TrainerMonitor] = None,
        callback: Optional[TrainerCallback] = None,
        metrics: Optional[MetricsProtocol] = None,
        workplace: str = "_logs",
        metric_log_file: str = "metrics.txt",
        rank: Optional[int] = None,
    ):
        self.config = config or {}
        self.state_config = state_config or {}
        self.num_epoch = num_epoch
        self.valid_portion = valid_portion
        self.use_amp = amp
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
        self.clip_norm = clip_norm
        self.monitor = monitor or TrainerMonitor()
        self.callback = callback or TrainerCallback()
        self.metrics = metrics
        self.ddp = rank is not None
        self.is_rank_0 = self.callback.is_rank_0 = rank is None or rank == 0
        # initialize artifact structure
        if self.is_rank_0:
            self.workplace = os.path.join(workplace, timestamp())
            if os.path.isdir(self.workplace):
                print(f"{WARNING_PREFIX}workplace already exists, it will be erased")
                shutil.rmtree(self.workplace)
            os.makedirs(self.workplace)
            self.metric_log_path = os.path.join(self.workplace, metric_log_file)
            with open(self.metric_log_path, "w"):
                pass
            self.checkpoint_folder = os.path.join(self.workplace, "checkpoints")
            os.makedirs(self.checkpoint_folder)
        # properties
        self.checkpoint_scores: Dict[str, float] = {}

    @property
    def validation_loader(self) -> DataLoaderProtocol:
        return self.valid_loader or self.train_loader_copy

    # init

    def _define_optimizer(
        self,
        params_name: str,
        optimizer_base: Type[Optimizer],
        optimizer_config: Dict[str, Any],
    ) -> Optimizer:
        if params_name == "all":
            parameters = self.model.parameters()
        else:
            attr = getattr(self.model, params_name)
            if not isinstance(attr, torch.nn.Module):
                parameters = attr
            else:
                parameters = attr.parameters()
        opt = optimizer_base(parameters, **optimizer_config)
        self.optimizers[params_name] = opt
        return opt

    def _init_optimizers(self) -> None:
        optimizers_settings = self.config.setdefault(
            "optimizers",
            {
                "all": {
                    "optimizer": "adam",
                    "optimizer_config": {},
                    "scheduler": None,
                    "scheduler_config": {},
                },
            },
        )
        self.optimizers: Dict[str, Optimizer] = {}
        self.schedulers: Dict[str, Optional[_LRScheduler]] = {}
        for params_name, opt_setting in optimizers_settings.items():
            optimizer = opt_setting["optimizer"]
            optimizer_config = opt_setting["optimizer_config"]
            scheduler = opt_setting["scheduler"]
            scheduler_config = opt_setting["scheduler_config"]
            optimizer_base = optimizer_dict[optimizer]
            opt = self._define_optimizer(params_name, optimizer_base, optimizer_config)
            if scheduler is None:
                self.schedulers[params_name] = None
            else:
                scheduler_base = scheduler_dict[scheduler]
                self.schedulers[params_name] = scheduler_base(opt, **scheduler_config)

    # core

    def _clip_norm_step(self) -> None:
        for opt in self.optimizers.values():
            self.grad_scaler.unscale_(opt)
        self._gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.clip_norm,
        )

    def _optimizer_step(self) -> None:
        for opt in self.optimizers.values():
            self.grad_scaler.step(opt)
            self.grad_scaler.update()
        for param in self.model.parameters():
            param.grad = None

    def _scheduler_step(self) -> None:
        for key, scheduler in self.schedulers.items():
            if scheduler is not None:
                if self.state.should_log_lr:
                    self.callback.log_lr(
                        f"lr-{key}",
                        scheduler.get_last_lr()[0],
                        self.state.step,
                    )
                scheduler.step()

    def _monitor_step(self) -> MonitorResults:
        outputs = None
        metric_outputs = None
        terminate = False
        save_checkpoint = False
        if self.state.should_monitor:
            # get metrics
            outputs, metric_outputs = self.get_metrics(portion=self.valid_portion)
            # logging
            if self.is_rank_0:
                self.callback.log_metrics(metric_outputs)
                if self.state.should_log_artifacts:
                    self.callback.log_artifacts(self)
                if self.state.should_log_metrics_msg:
                    self.callback.log_metrics_msg(
                        self.state,
                        metric_outputs,
                        self.metric_log_path,
                    )
            # check terminate
            if self.state.should_start_snapshot:
                score = metric_outputs.final_score
                if self.monitor.snapshot(score):
                    save_checkpoint = True
                if self.monitor.check_terminate(score):
                    terminate = True
        return MonitorResults(terminate, save_checkpoint, outputs, metric_outputs)

    def _step(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
    ) -> StepOutputs:
        batch = to_device(batch, self.device)
        # forward & loss
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            forward_results = self.model(batch_idx, batch, self.state)
            loss_dict = self.loss(forward_results, batch)
        # backward
        loss = loss_dict[LOSS_KEY]
        self.grad_scaler.scale(loss).backward()
        # clip norm
        if self.clip_norm > 0.0:
            self._clip_norm_step()
        # optimize
        self._optimizer_step()
        self._scheduler_step()
        return StepOutputs(forward_results, loss_dict)

    # api

    def fit(
        self,
        loss: LossProtocol,
        model: ModelProtocol,
        inference: InferenceProtocol,
        train_loader: DataLoaderProtocol,
        valid_loader: Optional[DataLoaderProtocol],
        *,
        cuda: Optional[str] = None,
    ) -> None:
        if self.is_rank_0:
            with open(os.path.join(self.workplace, "model.txt"), "w") as f:
                f.write(str(model))
        self.device = torch.device("cpu" if cuda is None else f"cuda:{cuda}")
        self.loss = loss
        self.model = model.to(self.device)
        self.inference = inference
        self.train_loader = train_loader
        self.train_loader_copy = copy.deepcopy(train_loader)
        self.valid_loader = valid_loader
        self.state = TrainerState(
            train_loader,
            num_epoch=self.num_epoch,
            **self.state_config,
        )
        # optimizer
        self._init_optimizers()
        # train
        has_ckpt = terminate = False
        print(f"{INFO_PREFIX}entered training loop")
        while self.state.should_train:
            try:
                self.state.epoch += 1
                for i, batch in enumerate(self.train_loader):
                    self.state.step += 1
                    step_outputs = self._step(i, batch)
                    self.callback.after_step(step_outputs)
                    monitor_results = self._monitor_step()
                    self.callback.after_monitor(monitor_results)
                    if self.is_rank_0 and monitor_results.save_checkpoint:
                        metric_outputs = monitor_results.metric_outputs
                        assert metric_outputs is not None
                        self.save_checkpoint(metric_outputs.final_score)
                    terminate = monitor_results.terminate
                    if terminate:
                        break
            except KeyboardInterrupt:
                print(f"{ERROR_PREFIX}keyboard interrupted")
                terminate = True
            if terminate:
                break
        # restore
        if not self.ddp and os.path.isdir(self.checkpoint_folder):
            print(f"{INFO_PREFIX}rolling back to the best checkpoint")
            has_ckpt = self.restore_checkpoint()
        # finalize
        self.state.set_terminate()
        if self.is_rank_0:
            _, self.final_results = self.get_metrics()
            self.callback.log_metrics_msg(
                self.state,
                self.final_results,
                self.metric_log_path,
            )
            if not has_ckpt:
                self.save_checkpoint(self.final_results.final_score)

    def get_metrics(
        self,
        *,
        portion: float = 1.0,
        loader: Optional[DataLoaderProtocol] = None,
    ) -> Tuple[InferenceOutputs, MetricsOutputs]:
        if loader is None:
            loader = self.validation_loader
        loss = self.loss if self.metrics is None else None
        outputs = self.inference.get_outputs(
            self.device,
            loader,
            portion=portion,
            state=self.state,
            loss=loss,
        )
        if self.metrics is not None:
            return outputs, self.metrics.evaluate(outputs)
        loss_items = outputs.loss_items
        assert loss_items is not None
        return outputs, MetricsOutputs(-loss_items[LOSS_KEY], loss_items)

    # checkpointing

    @staticmethod
    def _get_sorted_checkpoints(scores_path: str) -> List[str]:
        if not os.path.isfile(scores_path):
            return []
        with open(scores_path, "r") as f:
            scores = json.load(f)
        files = list(scores.keys())
        scores_list = [scores[file] for file in files]
        sorted_indices = np.argsort(scores_list)[::-1]
        return [files[i] for i in sorted_indices]

    def save_checkpoint(self, score: float, folder: Optional[str] = None) -> None:
        if folder is None:
            folder = self.checkpoint_folder
        scores_path = os.path.join(folder, SCORES_FILE)
        # leave top_k snapshots only
        if self.state.max_snapshot_file > 0:
            # better checkpoints will be placed earlier,
            #  which means `checkpoints[0]` is the best checkpoint
            checkpoints = self._get_sorted_checkpoints(scores_path)
            if len(checkpoints) >= self.state.max_snapshot_file:
                for file in checkpoints[self.state.max_snapshot_file - 1 :]:
                    self.checkpoint_scores.pop(file)
                    os.remove(os.path.join(folder, file))
        # pt
        file = f"{PT_PREFIX}{self.state.epoch}.pt"
        torch.save(self.model.state_dict(), os.path.join(folder, file))
        # scores
        self.checkpoint_scores[file] = score
        with open(scores_path, "w") as f:
            json.dump(self.checkpoint_scores, f)

    def restore_checkpoint(
        self,
        folder: str = None,
        strict: bool = True,
        state_dict_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> bool:
        if folder is None:
            folder = self.checkpoint_folder
        checkpoints = self._get_sorted_checkpoints(os.path.join(folder, SCORES_FILE))
        if not checkpoints:
            print(f"{WARNING_PREFIX}no model file found in {folder}")
            return False
        success = False
        for checkpoint in checkpoints:
            model_file = os.path.join(folder, checkpoint)
            if not os.path.isfile(model_file):
                continue
            print(f"{INFO_PREFIX}restoring from {model_file}")
            states = torch.load(model_file, map_location=self.device)
            if state_dict_callback is not None:
                state_dict_callback(states)
            self.model.load_state_dict(states, strict)
            success = True
            break
        return success


__all__ = [
    "Trainer",
    "TrainerCallback",
]
