# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Knowledge distillation training engine."""

from __future__ import annotations

import io
import math
import time
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import distributed as dist
from torch import nn

from ultralytics import __version__
from ultralytics.cfg import DEFAULT_CFG
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import GIT, LOCAL_RANK, LOGGER, RANK, TQDM, YAML, colorstr
from ultralytics.utils.torch_utils import (
    TORCH_2_4,
    EarlyStopping,
    ModelEMA,
    attempt_compile,
    autocast,
    convert_optimizer_state_dict_to_fp16,
    one_cycle,
    select_device,
    unset_deterministic,
    unwrap_model,
)


class _FeatureHook:
    """Callable hook that appends layer output to a storage list. Picklable unlike lambdas."""

    def __init__(self, storage):
        self.storage = storage

    def __call__(self, module, input, output):
        self.storage.append(output)


class DistillationWrapper(nn.Module):
    """Wraps student model + aligner into a single nn.Module for unified optimizer traversal."""

    def __init__(self, student, aligner):
        """Initialize DistillationWrapper.

        Args:
            student (nn.Module): Student model.
            aligner (nn.Module): Multi-scale feature aligner.
        """
        super().__init__()
        self.student = student
        self.aligner = aligner

    def __getattr__(self, name):
        """Delegate attribute access to student for unknown attributes."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.student, name)

    def forward(self, *args, **kwargs):
        """Forward pass delegates to student."""
        return self.student(*args, **kwargs)

    def loss(self, batch, preds=None):
        """Compute loss via student."""
        return self.student.loss(batch, preds)

    def init_criterion(self):
        """Initialize criterion via student."""
        return self.student.init_criterion()


def create_distiller(trainer_cls):
    """Factory function: creates a Distiller class inheriting from the given trainer.

    Args:
        trainer_cls: Base trainer class (e.g., DetectionTrainer, SegmentationTrainer).

    Returns:
        Distiller class that adds knowledge distillation to the base trainer.

    Example:
        >>> from ultralytics.models.yolo.detect.train import DetectionTrainer
        >>> Distiller = create_distiller(DetectionTrainer)
        >>> trainer = Distiller(overrides={"model": "yolov8n.yaml", "data": "coco8.yaml",
        ...     "distill_cfg": "ultralytics/cfg/distill_cfg.yaml"})
        >>> trainer.train()
    """

    class Distiller(trainer_cls):
        """Knowledge distillation trainer. Adds feature-level KD loss to the base task trainer."""

        def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
            """Initialize Distiller and load distillation config."""
            super().__init__(cfg, overrides, _callbacks)
            self.distill_cfg = self._load_distill_cfg()

        def _load_distill_cfg(self):
            """Load distillation config YAML. Returns None if KD is disabled."""
            if not getattr(self.args, "distill_cfg", None):
                return None
            return YAML.load(self.args.distill_cfg)

        # --- Teacher ---

        def _setup_teacher(self, model_path):
            """Load teacher model, move to device, freeze all parameters, set eval mode."""
            weights, _ = load_checkpoint(model_path)
            self.teacher = weights.to(self.device).eval()
            for p in self.teacher.parameters():
                p.requires_grad_(False)
            LOGGER.info(f"Teacher model loaded from {model_path} and frozen")

        # --- Module resolution & hooks ---

        def _resolve_module(self, model, spec):
            """Resolve a layer spec to an nn.Module.

            Args:
                model: The model to resolve from.
                spec: int (model.model[spec]) or str (model.get_submodule(spec)).
            """
            if isinstance(spec, int):
                return model.model[spec]
            return model.get_submodule(spec)

        def _register_feature_hooks(self, model, layer_specs, storage):
            """Register forward hooks on specified layers to capture output features.

            Returns:
                list: Hook handles for later removal.
            """
            handles = []
            hook_fn = _FeatureHook(storage)
            for spec in layer_specs:
                module = self._resolve_module(model, spec)
                h = module.register_forward_hook(hook_fn)
                handles.append(h)
            return handles

        # --- Channel extraction ---

        def _get_head_input_indices(self, model):
            """Get default distillation points: head input layer indices."""
            return model.model[-1].f

        def _get_layer_channels(self, model, layer_specs):
            """Measure output channels of specified layers via dummy forward pass."""
            temp_storage = []
            temp_hooks = []
            hook_fn = _FeatureHook(temp_storage)
            for spec in layer_specs:
                module = self._resolve_module(model, spec)
                h = module.register_forward_hook(hook_fn)
                temp_hooks.append(h)

            imgsz = self.args.imgsz
            dummy = torch.zeros(1, 3, imgsz, imgsz, device=self.device)
            with torch.no_grad():
                model(dummy)

            channels = [feat.shape[1] for feat in temp_storage]
            for h in temp_hooks:
                h.remove()
            return channels

        # --- Aligner & KD loss setup ---

        def _setup_aligner(self, student_channels, teacher_channels):
            """Create MultiScaleAligner from student/teacher channel info."""
            from ultralytics.nn.modules.aligner import ConvAligner, MultiScaleAligner

            aligner_name = self.distill_cfg.get("aligner", "ConvAligner")
            aligner_map = {"ConvAligner": ConvAligner}
            self.aligner_module = MultiScaleAligner(
                student_channels,
                teacher_channels,
                aligner_cls=aligner_map[aligner_name],
            ).to(self.device)

        def _kd_progress_string(self):
            """Return progress string header with kd_loss column inserted before Instances."""
            return ("\n" + "%11s" * (5 + len(self.loss_names))) % (
                "Epoch",
                "GPU_mem",
                *self.loss_names,
                "kd_loss",
                "Instances",
                "Size",
            )

        def _setup_kd_loss(self):
            """Initialize KD feature loss function."""
            from ultralytics.utils.loss import KDFeatureLoss

            loss_name = self.distill_cfg.get("loss", "mse")
            loss_map = {"mse": nn.MSELoss(reduction="mean")}
            self.kd_loss_fn = KDFeatureLoss(loss_fn=loss_map.get(loss_name))

        # --- Training setup override ---

        def _setup_train(self):
            """Override: add teacher, aligner, hooks, and wrapper on top of base trainer setup."""
            if self.distill_cfg is None:
                return super()._setup_train()

            # 1. Load student model
            ckpt = self.setup_model()
            self.model = self.model.to(self.device)
            self.set_model_attributes()

            # 2. Read distill config
            teacher_cfg = self.distill_cfg["teacher"]
            student_cfg = self.distill_cfg["student"]
            self.kd_weight = self.distill_cfg.get("weight", 1.0)

            # 3. Load & freeze teacher
            self._setup_teacher(teacher_cfg["model"])

            # 4. Determine distillation points
            teacher_layers = teacher_cfg.get("layers") or self._get_head_input_indices(self.teacher)
            student_layers = student_cfg.get("layers") or self._get_head_input_indices(self.model)
            assert len(teacher_layers) == len(student_layers), (
                f"teacher ({len(teacher_layers)}) and student ({len(student_layers)}) layer count mismatch"
            )

            # 5. Measure channels via dummy forward
            teacher_channels = self._get_layer_channels(self.teacher, teacher_layers)
            student_channels = self._get_layer_channels(self.model, student_layers)
            LOGGER.info(f"KD student channels: {student_channels}, teacher channels: {teacher_channels}")

            # 6. Create aligner
            self._setup_aligner(student_channels, teacher_channels)

            # 7. Register feature hooks
            self._teacher_feats = []
            self._student_feats = []
            self._hooks = []
            self._hooks += self._register_feature_hooks(self.teacher, teacher_layers, self._teacher_feats)
            self._hooks += self._register_feature_hooks(self.model, student_layers, self._student_feats)

            # 8. Wrap student + aligner
            self.model = DistillationWrapper(self.model, self.aligner_module)

            # 9. KD loss
            self._setup_kd_loss()

            # 10. Remaining base setup: compile, freeze, AMP, DDP, batch, optimizer, EMA, etc.
            self.model = attempt_compile(self.model, device=self.device, mode=self.args.compile)

            freeze_list = (
                self.args.freeze
                if isinstance(self.args.freeze, list)
                else range(self.args.freeze) if isinstance(self.args.freeze, int) else []
            )
            always_freeze_names = [".dfl"]
            freeze_layer_names = [f"model.{x}." for x in freeze_list] + always_freeze_names
            self.freeze_layer_names = freeze_layer_names
            for k, v in self.model.named_parameters():
                if any(x in k for x in freeze_layer_names):
                    LOGGER.info(f"Freezing layer '{k}'")
                    v.requires_grad = False
                elif not v.requires_grad and v.dtype.is_floating_point:
                    LOGGER.warning(
                        f"setting 'requires_grad=True' for frozen layer '{k}'. "
                        "See ultralytics.engine.trainer for customization of frozen layers."
                    )
                    v.requires_grad = True

            from ultralytics.utils.checks import check_amp, check_imgsz

            self.amp = torch.tensor(self.args.amp).to(self.device)
            if self.amp and RANK in {-1, 0}:
                from ultralytics.utils import callbacks

                callbacks_backup = callbacks.default_callbacks.copy()
                self.amp = torch.tensor(check_amp(self.model), device=self.device)
                callbacks.default_callbacks = callbacks_backup
            if RANK > -1 and self.world_size > 1:
                dist.broadcast(self.amp.int(), src=0)
            self.amp = bool(self.amp)
            self.scaler = (
                torch.amp.GradScaler("cuda", enabled=self.amp)
                if TORCH_2_4
                else torch.cuda.amp.GradScaler(enabled=self.amp)
            )
            if self.world_size > 1:
                self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[RANK], find_unused_parameters=True)

            gs = max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32)
            self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)
            self.stride = gs

            if self.batch_size < 1 and RANK == -1:
                self.args.batch = self.batch_size = self.auto_batch()

            self._build_train_pipeline()
            self.validator = self.get_validator()
            self.tkd_loss = None
            self.ema = ModelEMA(self.model)
            if RANK in {-1, 0}:
                metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix="val")
                self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
                if self.args.plots:
                    self.plot_training_labels()

            self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
            self.resume_training(ckpt)
            self.scheduler.last_epoch = self.start_epoch - 1
            self.run_callbacks("on_pretrain_routine_end")

        # --- Training loop override ---

        def _do_train(self):
            """Override: same as base _do_train but with KD forward logic."""
            if self.distill_cfg is None:
                return super()._do_train()

            if self.world_size > 1:
                self._setup_ddp()
            self._setup_train()

            nb = len(self.train_loader)
            nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1
            last_opt_step = -1
            self.epoch_time = None
            self.epoch_time_start = time.time()
            self.train_time_start = time.time()
            self.run_callbacks("on_train_start")
            LOGGER.info(
                f"Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n"
                f"Using {self.train_loader.num_workers * (self.world_size or 1)} dataloader workers\n"
                f"Logging results to {colorstr('bold', self.save_dir)}\n"
                f"Starting training for "
                + (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
            )
            if self.args.close_mosaic:
                base_idx = (self.epochs - self.args.close_mosaic) * nb
                self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
            epoch = self.start_epoch
            self.optimizer.zero_grad()
            self._oom_retries = 0
            while True:
                self.epoch = epoch
                self.run_callbacks("on_train_epoch_start")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.scheduler.step()

                self._model_train()
                # Keep teacher in eval mode
                self.teacher.eval()

                if RANK != -1:
                    self.train_loader.sampler.set_epoch(epoch)
                pbar = enumerate(self.train_loader)
                if epoch == (self.epochs - self.args.close_mosaic):
                    self._close_dataloader_mosaic()
                    self.train_loader.reset()

                if RANK in {-1, 0}:
                    LOGGER.info(self._kd_progress_string())
                    pbar = TQDM(enumerate(self.train_loader), total=nb)
                self.tloss = None
                self.tkd_loss = None
                for i, batch in pbar:
                    self.run_callbacks("on_train_batch_start")
                    ni = i + nb * epoch
                    if ni <= nw:
                        xi = [0, nw]
                        self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                        for x in self.optimizer.param_groups:
                            x["lr"] = np.interp(
                                ni,
                                xi,
                                [
                                    self.args.warmup_bias_lr if x.get("param_group") == "bias" else 0.0,
                                    x["initial_lr"] * self.lf(epoch),
                                ],
                            )
                            if "momentum" in x:
                                x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                    # === KD Forward ===
                    try:
                        with autocast(self.amp):
                            batch = self.preprocess_batch(batch)

                            # Teacher forward (no grad) - hooks capture features
                            self._teacher_feats.clear()
                            with torch.no_grad():
                                self.teacher(batch["img"])

                            # Student forward - hooks capture features + task loss
                            self._student_feats.clear()
                            if self.args.compile:
                                preds = self.model(batch["img"])
                                loss, self.loss_items = unwrap_model(self.model).loss(batch, preds)
                            else:
                                loss, self.loss_items = self.model(batch)

                            # Align student features and compute KD loss
                            aligned_feats = unwrap_model(self.model).aligner(self._student_feats)
                            kd_loss = self.kd_loss_fn(aligned_feats, self._teacher_feats)
                            self.loss = loss.sum() + self.kd_weight * kd_loss * batch["img"].shape[0]

                            if RANK != -1:
                                self.loss *= self.world_size
                            self.tloss = (
                                self.loss_items if self.tloss is None else (self.tloss * i + self.loss_items) / (i + 1)
                            )
                            kd_val = kd_loss.detach().item()
                            self.tkd_loss = kd_val if self.tkd_loss is None else (self.tkd_loss * i + kd_val) / (i + 1)

                        # Backward
                        self.scaler.scale(self.loss).backward()
                    except torch.cuda.OutOfMemoryError:
                        if epoch > self.start_epoch or self._oom_retries >= 3 or RANK != -1:
                            raise
                        self._oom_retries += 1
                        old_batch = self.batch_size
                        self.args.batch = self.batch_size = max(self.batch_size // 2, 1)
                        LOGGER.warning(
                            f"CUDA out of memory with batch={old_batch}. "
                            f"Reducing to batch={self.batch_size} and retrying ({self._oom_retries}/3)."
                        )
                        self._clear_memory()
                        self._build_train_pipeline()
                        self.scheduler.last_epoch = self.start_epoch - 1
                        nb = len(self.train_loader)
                        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1
                        last_opt_step = -1
                        self.optimizer.zero_grad()
                        break
                    if ni - last_opt_step >= self.accumulate:
                        self.optimizer_step()
                        last_opt_step = ni

                        if self.args.time:
                            self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                            if RANK != -1:
                                broadcast_list = [self.stop if RANK == 0 else None]
                                dist.broadcast_object_list(broadcast_list, 0)
                                self.stop = broadcast_list[0]
                            if self.stop:
                                break

                    # Log
                    if RANK in {-1, 0}:
                        loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
                        pbar.set_description(
                            ("%11s" * 2 + "%11.4g" * (3 + loss_length))
                            % (
                                f"{epoch + 1}/{self.epochs}",
                                f"{self._get_memory():.3g}G",
                                *(self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)),
                                self.tkd_loss or 0,
                                batch["cls"].shape[0],
                                batch["img"].shape[-1],
                            )
                        )
                        self.run_callbacks("on_batch_end")
                        if self.args.plots and ni in self.plot_idx:
                            self.plot_training_samples(batch, ni)

                    self.run_callbacks("on_train_batch_end")
                    if self.stop:
                        break
                else:
                    self._oom_retries = 0

                if self._oom_retries and not self.stop:
                    continue

                if hasattr(unwrap_model(self.model).criterion, "update"):
                    unwrap_model(self.model).criterion.update()

                self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}

                self.run_callbacks("on_train_epoch_end")
                if RANK in {-1, 0}:
                    self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

                # Validation
                final_epoch = epoch + 1 >= self.epochs
                if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                    self._clear_memory(None if self.device.type == "mps" else 0.5)
                    self.metrics, self.fitness = self.validate()

                # NaN recovery
                if self._handle_nan_recovery(epoch):
                    continue

                self.nan_recovery_attempts = 0
                if RANK in {-1, 0}:
                    kd_metrics = {"train/kd_loss": round(self.tkd_loss, 5)} if self.tkd_loss is not None else {}
                    self.save_metrics(metrics={**self.label_loss_items(self.tloss), **kd_metrics, **self.metrics, **self.lr})
                    self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                    if self.args.time:
                        self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)

                    if self.args.save or final_epoch:
                        self.save_model()
                        self.run_callbacks("on_model_save")

                t = time.time()
                self.epoch_time = t - self.epoch_time_start
                self.epoch_time_start = t
                if self.args.time:
                    mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                    self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                    self._setup_scheduler()
                    self.scheduler.last_epoch = self.epoch
                    self.stop |= epoch >= self.epochs
                self.run_callbacks("on_fit_epoch_end")
                self._clear_memory(None if self.device.type == "mps" else 0.5)

                if RANK != -1:
                    broadcast_list = [self.stop if RANK == 0 else None]
                    dist.broadcast_object_list(broadcast_list, 0)
                    self.stop = broadcast_list[0]
                if self.stop:
                    break
                epoch += 1

            seconds = time.time() - self.train_time_start
            LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.")
            self.final_eval()
            if RANK in {-1, 0}:
                if self.args.plots:
                    self.plot_metrics()
                self.run_callbacks("on_train_end")
            self._clear_memory()
            unset_deterministic()
            self.run_callbacks("teardown")

        # --- Checkpoint saving override ---

        def save_model(self):
            """Save student as standard YOLO checkpoint + aligner separately."""
            if self.distill_cfg is None:
                return super().save_model()

            import io
            from datetime import datetime

            # Extract student from EMA (unwrap DistillationWrapper)
            ema_model = unwrap_model(self.ema.ema)
            if isinstance(ema_model, DistillationWrapper):
                # Clear hooks before deepcopy (EMA hooks are unused copies from training model)
                for module in ema_model.student.modules():
                    module._forward_hooks.clear()
                student_ema = deepcopy(ema_model.student).half()
                aligner_ema = deepcopy(ema_model.aligner).half()
            else:
                student_ema = deepcopy(ema_model).half()
                aligner_ema = None

            # Save standard YOLO checkpoint (student only)
            buffer = io.BytesIO()
            torch.save(
                {
                    "epoch": self.epoch,
                    "best_fitness": self.best_fitness,
                    "model": None,
                    "ema": student_ema,
                    "updates": self.ema.updates,
                    "optimizer": convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
                    "scaler": self.scaler.state_dict(),
                    "train_args": vars(self.args),
                    "train_metrics": {**self.metrics, **{"fitness": self.fitness}},
                    "train_results": self.read_results_csv(),
                    "date": datetime.now().isoformat(),
                    "version": __version__,
                    "git": {
                        "root": str(GIT.root),
                        "branch": GIT.branch,
                        "commit": GIT.commit,
                        "origin": GIT.origin,
                    },
                    "license": "AGPL-3.0 (https://ultralytics.com/license)",
                    "docs": "https://docs.ultralytics.com",
                },
                buffer,
            )
            serialized_ckpt = buffer.getvalue()

            self.wdir.mkdir(parents=True, exist_ok=True)
            self.last.write_bytes(serialized_ckpt)
            if self.best_fitness == self.fitness:
                self.best.write_bytes(serialized_ckpt)
            if (self.save_period > 0) and (self.epoch % self.save_period == 0):
                (self.wdir / f"epoch{self.epoch}.pt").write_bytes(serialized_ckpt)

            # Save aligner separately
            if aligner_ema is not None:
                torch.save({"aligner": aligner_ema, "epoch": self.epoch}, self.wdir / "aligner_last.pt")

    Distiller.__name__ = "Distiller"
    Distiller.__qualname__ = "Distiller"
    return Distiller
