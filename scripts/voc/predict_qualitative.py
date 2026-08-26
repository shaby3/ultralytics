# python scripts/voc/predict_qualitative.py
"""정성 비교 장표용 — student / teacher / KD 최종본을 같은 이미지에 돌린다.

`val/` 에 남는 val_batch*_pred.jpg 는 이 용도로 못 쓴다. 두 가지 이유다 (README §7.7):
  1. 배치 크기가 런마다 다르고(n 32, s 16, m 8) rect=True 가 종횡비로 정렬해서,
     같은 val_batch0 이라도 모델마다 다른 이미지 묶음이다.
  2. conf=0.001 · max_det=300 으로 그려져 신뢰도 0.1% 박스까지 깔린다. mAP 계산엔 맞지만
     "무엇을 더 잡았나" 를 보여주는 그림은 아니다.

그래서 predict 를 conf=0.25 로 다시 돌린다. 두 패스다.
  1패스: test2007 전체를 3모델로 추론해 이미지별 TP/FN/FP 를 센다 → ranking.csv
  2패스: 상위 K장만 다시 추론해 GT/student/teacher/KD 4패널을 붙인다

결과: runs/detect/voc/qualitative/
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import yaml

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

REPO = Path(__file__).resolve().parents[2]
IMAGES = Path("C:/Users/SSAFY/datasets/VOC/images/test2007")
LABELS = Path("C:/Users/SSAFY/datasets/VOC/labels/test2007")
OUT = REPO / "runs/detect/voc/qualitative"

# 비교 대상 3종. 라벨은 패널 제목에 그대로 박힌다.
MODELS = [
    ("student", "yolov8n (baseline)", REPO / "runs/detect/voc/baseline/yolov8n/train/weights/best.pt"),
    ("teacher", "yolov8s (teacher)", REPO / "runs/detect/voc/baseline/yolov8s/train/weights/best.pt"),
    ("kd", "yolov8n + KD (QMSE)", REPO / "runs/detect/voc/kd_head1/yolov8n_from_8s_coco_qmse/train/weights/best.pt"),
]

# val 과 NMS 조건은 맞추고 conf 만 올린다 — predict 기본값
PREDICT_ARGS = dict(imgsz=640, conf=0.25, iou=0.7, max_det=300, device=0, half=False, verbose=False)

IOU_THR = 0.5  # TP 판정 기준


def load_names():
    """VOC.yaml 의 클래스 이름 (id -> name)."""
    cfg = yaml.safe_load((REPO / "ultralytics/cfg/datasets/VOC.yaml").read_text(encoding="utf-8"))
    return cfg["names"]


def load_gt(stem, w, h):
    """YOLO 정규화 라벨을 픽셀 xyxy 로. 반환: (N,4) float, (N,) int."""
    path = LABELS / f"{stem}.txt"
    if not path.exists():
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=int)
    boxes, cls = [], []
    for line in path.read_text().strip().splitlines():
        if not line.strip():
            continue
        c, cx, cy, bw, bh = line.split()[:5]
        cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
        boxes.append([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
        cls.append(int(c))
    return np.array(boxes, dtype=np.float32), np.array(cls, dtype=int)


def iou_matrix(a, b):
    """(N,4) x (M,4) -> (N,M) IoU."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def count_hits(pred_boxes, pred_cls, pred_conf, gt_boxes, gt_cls):
    """신뢰도 내림차순 greedy 매칭. 반환: (tp, fp, fn).

    val 의 mAP 계산과는 다른 지표다 — 여기서는 conf=0.25 한 지점의 셈이고,
    "어떤 이미지를 장표에 올릴까" 를 고르는 용도로만 쓴다.
    """
    order = np.argsort(-pred_conf)
    used = np.zeros(len(gt_boxes), dtype=bool)
    ious = iou_matrix(pred_boxes[order] if len(pred_boxes) else pred_boxes, gt_boxes)
    tp = 0
    for i, pi in enumerate(order):
        cand = np.where((~used) & (gt_cls == pred_cls[pi]) & (ious[i] >= IOU_THR))[0]
        if len(cand):
            best = cand[np.argmax(ious[i][cand])]
            used[best] = True
            tp += 1
    return tp, len(pred_boxes) - tp, int((~used).sum())


def write_source_list(paths, name):
    """경로 리스트를 .txt 로 넘긴다.

    파이썬 list 를 source 로 주면 autocast_list 가 PIL 로 미리 읽어버려서 r.path 가
    'image0' 이 된다 (loaders.py autocast_list). .txt 는 파일 경로가 그대로 남는다.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    txt = OUT / name
    txt.write_text("\n".join(str(Path(p).absolute()) for p in paths), encoding="utf-8")
    return str(txt)


def draw_gt(img, boxes, cls, names):
    """GT 패널 — predict 결과와 같은 Annotator 로 그려야 선 굵기가 맞는다."""
    ann = Annotator(img.copy(), line_width=None, example=str(names))
    for box, c in zip(boxes, cls):
        ann.box_label(box, names[int(c)], color=colors(int(c), True))
    return ann.result()


def add_title(img, text):
    """패널 위에 검은 제목 바를 얹는다."""
    h, w = img.shape[:2]
    bar = int(max(28, h * 0.075))
    scale = bar / 40
    canvas = np.zeros((h + bar, w, 3), dtype=np.uint8)
    canvas[bar:] = img
    cv2.putText(canvas, text, (8, int(bar * 0.72)), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), max(1, int(scale * 2)))
    return canvas


def score_all(models, names, limit):
    """1패스 — 전체 이미지에 대해 모델별 TP/FP/FN 을 센다."""
    files = sorted(IMAGES.glob("*.jpg"))
    if limit:
        files = files[:limit]
    print(f"[1/2] scoring {len(files)} images x {len(models)} models")

    stats = {f.stem: {} for f in files}
    sources = write_source_list(files, "_score_sources.txt")
    for key, label, weights in models:
        model = YOLO(weights)
        print(f"  - {key}: {weights.name}")
        for r in model.predict(source=sources, stream=True, **PREDICT_ARGS):
            stem = Path(r.path).stem
            h, w = r.orig_shape
            gt_boxes, gt_cls = load_gt(stem, w, h)
            b = r.boxes
            tp, fp, fn = count_hits(
                b.xyxy.cpu().numpy(), b.cls.cpu().numpy().astype(int), b.conf.cpu().numpy(), gt_boxes, gt_cls
            )
            stats[stem][key] = (tp, fp, fn, len(gt_cls))
        del model

    rows = []
    for stem, per in stats.items():
        if len(per) != len(models):
            continue
        s_tp, s_fp, s_fn, n_gt = per["student"]
        k_tp, k_fp, k_fn, _ = per["kd"]
        t_tp, t_fp, t_fn, _ = per["teacher"]
        rows.append(
            dict(
                image=stem,
                n_gt=n_gt,
                gain_recall=s_fn - k_fn,  # KD 가 추가로 잡아낸 GT 수
                gain_precision=s_fp - k_fp,  # KD 가 줄인 오검출 수
                student_tp=s_tp, student_fp=s_fp, student_fn=s_fn,
                teacher_tp=t_tp, teacher_fp=t_fp, teacher_fn=t_fn,
                kd_tp=k_tp, kd_fp=k_fp, kd_fn=k_fn,
            )
        )
    rows.sort(key=lambda r: (-r["gain_recall"], -r["gain_precision"], -r["n_gt"]))

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "ranking.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  -> {OUT / 'ranking.csv'}")
    return rows


def render(models, names, stems):
    """2패스 — 고른 이미지만 다시 추론해 4패널로 붙인다."""
    print(f"[2/2] rendering {len(stems)} images")
    panels_dir = OUT / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)
    sources = write_source_list([IMAGES / f"{s}.jpg" for s in stems], "_render_sources.txt")

    drawn = {s: {} for s in stems}
    for key, label, weights in models:
        model = YOLO(weights)
        for r in model.predict(source=sources, stream=True, **PREDICT_ARGS):
            stem = Path(r.path).stem
            drawn[stem][key] = r.plot()
        del model

    for stem in stems:
        img = cv2.imread(str(IMAGES / f"{stem}.jpg"))
        h, w = img.shape[:2]
        gt_boxes, gt_cls = load_gt(stem, w, h)
        cells = [("gt", "Ground Truth", draw_gt(img, gt_boxes, gt_cls, names))]
        cells += [(key, label, drawn[stem][key]) for key, label, _ in models]

        for key, label, cell in cells:
            cv2.imwrite(str(panels_dir / f"{stem}_{key}.jpg"), cell)

        titled = [add_title(c, t) for _, t, c in cells]
        grid = np.vstack([np.hstack(titled[:2]), np.hstack(titled[2:])])
        cv2.imwrite(str(OUT / f"{stem}_compare.jpg"), grid)
    print(f"  -> {OUT}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--topk", type=int, default=12, help="렌더링할 상위 이미지 수")
    p.add_argument("--limit", type=int, default=0, help="스코어링할 이미지 수 상한 (0=전체 4952장)")
    p.add_argument("--images", nargs="*", default=None, help="이미지 stem 직접 지정 (예: 000123). 주면 스코어링을 건너뛴다")
    args = p.parse_args()

    names = load_names()
    models = [(k, l, w) for k, l, w in MODELS]
    for _, _, w in models:
        if not w.exists():
            raise SystemExit(f"가중치 없음: {w}")

    if args.images:
        render(models, names, args.images)
        return

    rows = score_all(models, names, args.limit)
    picked = [r["image"] for r in rows[: args.topk]]
    print("  상위:", ", ".join(f"{r['image']}(+{r['gain_recall']})" for r in rows[: args.topk]))
    render(models, names, picked)


if __name__ == "__main__":
    main()
