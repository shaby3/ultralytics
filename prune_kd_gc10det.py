import gc
import multiprocessing
from copy import deepcopy
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.engine.distiller import create_distiller
from ultralytics.nn.modules import Detect
from ultralytics.nn.modules.block import Attention
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import RANK
import torch_pruning as tp

from convert_weights import convert

# ---------------------------------------------------------------------------
# Weight paths
# ---------------------------------------------------------------------------
STUDENT_SRC  = r'runs\detect\gc10_det\kd\kd_unified_yolo26n_gc10_det_200epoch\weights\best.pt'
STUDENT_CKPT = r'runs\detect\gc10_det\kd\kd_unified_yolo26n_gc10_det_200epoch\weights\best_converted.pt'
TEACHER_SRC  = r'runs\detect\gc10_det\train\baseline_yolo26m_gc10_det_200epoch\weights\best.pt'
TEACHER_CKPT = r'runs\detect\gc10_det\train\baseline_yolo26m_gc10_det_200epoch\weights\best_converted.pt'
DISTILL_CFG  = 'ultralytics/cfg/distill_cfg_pruned_gc10_det.yaml'


def _ensure_converted(src: str, dst: str, yaml_name: str) -> None:
    if Path(dst).exists():
        return
    print(f'[convert] {src} → {dst}')
    convert(src, dst, yaml_name)


# ---------------------------------------------------------------------------
# KD fine-tune helper
# ---------------------------------------------------------------------------

def _train_pruned_kd(yolo: YOLO, distill_cfg: str, **kwargs) -> None:
    overrides = yolo.overrides.copy()
    overrides.update(kwargs)
    overrides['mode'] = 'train'
    overrides['distill_cfg'] = distill_cfg
    overrides.setdefault('model', STUDENT_CKPT)

    trainer_cls = yolo.task_map[yolo.task]['trainer']
    Distiller   = create_distiller(trainer_cls)
    trainer     = Distiller(overrides=overrides, _callbacks=yolo.callbacks)
    # BaseTrainer.setup_model()은 self.model이 이미 nn.Module이면 로딩을 건너뛴다.
    # prune_gc10det.py와 동일한 방식으로 pruned 모델을 직접 주입.
    trainer.model = deepcopy(yolo.model)

    trainer.train()

    if RANK in {-1, 0}:
        ckpt_path = trainer.best if trainer.best.exists() else trainer.last
        yolo.model, _ = load_checkpoint(str(ckpt_path))
        yolo.model = yolo.model.float()
        yolo.overrides = yolo.model.args
    yolo.trainer = trainer


# ---------------------------------------------------------------------------
# Iterative pruning loop
# ---------------------------------------------------------------------------

def iterative_prune_kd(model: YOLO, distill_cfg: str, data, finetune_epochs,
                       target_prune_ratio, iterative_steps=1, imgsz=640,
                       batch=8, device=None, name='prune_kd', max_map_drop=0.10,
                       workers=2, project='./runs/detect'):
    per_step_ratio = 1 - (1 - target_prune_ratio) ** (1 / iterative_steps)

    init_map = model.val(data=data, imgsz=imgsz, batch=batch, device=device,
                         project=project, workers=workers,
                         name='val_init', split='test').box.map
    gc.collect()
    torch.cuda.empty_cache()

    # val() fuses layers; reload to get non-fused, grad-capable weights
    ckpt_path = model.ckpt_path
    model.model, _ = load_checkpoint(str(ckpt_path))
    model.model = model.model.float()

    for step in range(iterative_steps):
        model.model.float().train()
        for p in model.model.parameters():
            p.requires_grad_(True)

        ignored_layers = [m for m in model.model.modules()
                          if isinstance(m, (Detect, Attention))]

        example_inputs = torch.zeros(1, 3, imgsz, imgsz)
        pruner = tp.pruner.MagnitudePruner(
            model.model,
            example_inputs,
            importance=tp.importance.MagnitudeImportance(p=2),
            iterative_steps=1,
            pruning_ratio=per_step_ratio,
            ignored_layers=ignored_layers,
        )
        pruner.step()

        macs, params = tp.utils.count_ops_and_params(model.model, example_inputs)
        print(f'[Step {step}] MACs: {macs / 1e9:.2f}G, Params: {params / 1e6:.2f}M')

        del pruner
        gc.collect()
        torch.cuda.empty_cache()

        for p in model.model.parameters():
            p.requires_grad_(True)

        _train_pruned_kd(model, distill_cfg=distill_cfg,
                         name=f'{name}_step{step}', data=data,
                         epochs=finetune_epochs, imgsz=imgsz, batch=batch,
                         device=device, project=project, workers=workers)
        last_ckpt = str(model.trainer.best if model.trainer.best.exists() else model.trainer.last)
        model.trainer = None
        gc.collect()
        torch.cuda.empty_cache()

        current_map = model.val(data=data, imgsz=imgsz, batch=batch, device=device,
                                project=project, workers=workers,
                                name=f'val_step{step}', split='test').box.map
        gc.collect()
        torch.cuda.empty_cache()
        print(f'[Step {step}] mAP: {current_map:.4f} (init: {init_map:.4f})')

        if init_map - current_map > max_map_drop:
            print(f'Early stop: mAP drop {init_map - current_map:.4f} > {max_map_drop}')
            break

        # val() fuses layers; reload before next pruning step
        if step < iterative_steps - 1:
            model.model, _ = load_checkpoint(last_ckpt)
            model.model = model.model.float()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def prunetrain_kd(model: YOLO, distill_cfg: str, train_epochs, prune_epochs=0,
                  quick_pruning=True, prune_ratio=0.5, prune_iterative_steps=1,
                  data='coco.yaml', name='yolo11', imgsz=640, batch=8,
                  device=None, max_map_drop=0.10, workers=2,
                  project='./runs/detect'):
    finetune_epochs = prune_epochs if not quick_pruning else train_epochs

    if not quick_pruning:
        assert train_epochs > 0 and prune_epochs > 0
        model.train(data=data, epochs=train_epochs, imgsz=imgsz, batch=batch,
                    device=device, name=name, project=project, prune=False,
                    workers=workers)

    iterative_prune_kd(
        model, distill_cfg=distill_cfg, data=data,
        finetune_epochs=finetune_epochs,
        target_prune_ratio=prune_ratio,
        iterative_steps=prune_iterative_steps,
        imgsz=imgsz, batch=batch, device=device,
        name=name, max_map_drop=max_map_drop, workers=workers,
        project=project,
    )


if __name__ == '__main__':
    multiprocessing.freeze_support()

    _ensure_converted(STUDENT_SRC, STUDENT_CKPT, 'yolo26n.yaml')
    _ensure_converted(TEACHER_SRC, TEACHER_CKPT, 'yolo26m.yaml')

    model = YOLO(STUDENT_CKPT)

    prunetrain_kd(
        model,
        distill_cfg=DISTILL_CFG,
        quick_pruning=True,
        data='GC10-DET.yaml',
        train_epochs=100,
        prune_epochs=100,
        imgsz=640,
        batch=32,
        device=[0],
        name='yolo26n_gc10det_kd',
        prune_ratio=0.5,
        prune_iterative_steps=2,
        workers=4,
        project='gc10_det/pruning_kd',
    )
