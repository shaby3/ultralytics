import csv
import gc
import multiprocessing
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.nn.modules import Detect
from ultralytics.nn.modules.block import Attention
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import RANK
from ultralytics.utils.torch_utils import get_flops, get_num_params
import torch_pruning as tp

# ---------------------------------------------------------------------------
# Project root & weight paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = str(PROJECT_ROOT / 'runs' / 'detect' / 'voc' / 'train'
                 / 'baseline_yolo26n_100epoch' / 'weights' / 'best.pt')


# ---------------------------------------------------------------------------
# Metrics logging helper
# ---------------------------------------------------------------------------

def _save_metrics_row(csv_path: Path, row: dict) -> None:
    write_header = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"Metrics saved → {csv_path}")


# ---------------------------------------------------------------------------
# Fine-tune helper
# ---------------------------------------------------------------------------

def _train_pruned(yolo: YOLO, **kwargs) -> None:
    overrides = yolo.overrides.copy()
    overrides.update(kwargs)
    overrides['mode'] = 'train'

    trainer_cls = yolo.task_map[yolo.task]['trainer']
    trainer = trainer_cls(overrides=overrides, _callbacks=yolo.callbacks)
    trainer.model = yolo.model
    trainer.hub_session = getattr(yolo, 'session', None)
    trainer.prune = False
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

def iterative_prune(model: YOLO, data, finetune_epochs, target_prune_ratio,
                    iterative_steps=1, imgsz=640, batch=8, device=None,
                    name='prune', max_map_drop=0.10, workers=2,
                    project='./runs/detect', patience=0):
    per_step_ratio = 1 - (1 - target_prune_ratio) ** (1 / iterative_steps)

    params_M = round(get_num_params(model.model) / 1e6, 3)
    gflops   = round(get_flops(model.model, imgsz=imgsz), 1)
    results = model.val(data=data, imgsz=imgsz, batch=1, device=device,
                        project=project, workers=workers,
                        name='val_init', split='test')
    init_map = results.box.map
    csv_path = results.save_dir.parent / name / 'pruning_metrics.csv'
    inference_ms = results.speed['inference']
    _save_metrics_row(csv_path, {
        'stage':      'val_init',
        'params_M':   params_M,
        'GFLOPs':     gflops,
        'mAP50':      round(results.results_dict['metrics/mAP50(B)'], 4),
        'mAP50_95':   round(results.results_dict['metrics/mAP50-95(B)'], 4),
        'latency_ms': round(inference_ms, 2),
        'FPS':        round(1000 / inference_ms, 1),
    })
    gc.collect()
    torch.cuda.empty_cache()

    # val() fuses layers inside inference_mode; reload to get non-fused, grad-capable weights
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

        _train_pruned(model, name=f'{name}_pruned_step{step}', data=data,
                      epochs=finetune_epochs, imgsz=imgsz, batch=batch, device=device,
                      project=project, workers=workers, patience=patience)
        last_ckpt = str(model.trainer.best if model.trainer.best.exists() else model.trainer.last)
        model.trainer = None
        gc.collect()
        torch.cuda.empty_cache()

        params_M = round(get_num_params(model.model) / 1e6, 3)
        gflops   = round(get_flops(model.model, imgsz=imgsz), 1)
        results = model.val(data=data, imgsz=imgsz, batch=1, device=device,
                            project=project, workers=workers,
                            name=f'val_step{step}', split='test')
        current_map = results.box.map
        inference_ms = results.speed['inference']
        _save_metrics_row(csv_path, {
            'stage':      f'step{step}',
            'params_M':   params_M,
            'GFLOPs':     gflops,
            'mAP50':      round(results.results_dict['metrics/mAP50(B)'], 4),
            'mAP50_95':   round(results.results_dict['metrics/mAP50-95(B)'], 4),
            'latency_ms': round(inference_ms, 2),
            'FPS':        round(1000 / inference_ms, 1),
        })
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

def prunetrain(model: YOLO, train_epochs, prune_epochs=0, quick_pruning=True,
               prune_ratio=0.5, prune_iterative_steps=1, data='coco.yaml',
               name='yolo11', imgsz=640, batch=8, device=None,
               max_map_drop=0.10, workers=2, patience=0,
               project='./runs/detect'):
    finetune_epochs = prune_epochs if not quick_pruning else train_epochs

    if not quick_pruning:
        assert train_epochs > 0 and prune_epochs > 0
        model.train(data=data, epochs=train_epochs, imgsz=imgsz, batch=batch,
                    device=device, name=name, project=project, prune=False,
                    workers=workers)

    iterative_prune(
        model, data=data,
        finetune_epochs=finetune_epochs,
        target_prune_ratio=prune_ratio,
        iterative_steps=prune_iterative_steps,
        imgsz=imgsz, batch=batch, device=device,
        name=name, max_map_drop=max_map_drop, workers=workers,
        project=project, patience=patience,
    )


if __name__ == '__main__':
    multiprocessing.freeze_support()

    model = YOLO(MODEL_PATH)

    prunetrain(
        model,
        quick_pruning=True,         # Quick Pruning: 별도 초기 학습 없이 바로 pruning → fine-tune
        data='VOC.yaml',            # Dataset config
        train_epochs=100,           # 스텝당 최대 100 epoch, patience로 조기 종료
        prune_epochs=100,
        imgsz=640,
        batch=64,
        device=[0],
        name='yolo26n_voc_3step_patience30',
        prune_ratio=0.5,            # Pruning Ratio (50%)
        prune_iterative_steps=3,    # 3-step iterative pruning
        patience=30,                # 30 epoch 개선 없으면 조기 종료
        workers=2,
        project='voc/pruning/3step_patience',
    )
