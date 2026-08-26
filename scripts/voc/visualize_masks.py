# python scripts/voc/visualize_masks.py
"""AMSE(크기 가중) · QMSE(품질 가중) 마스크 시각화 + 집중도 실측.

README §4 의 "AMSE attention 분포 실측"·"QMSE 마스크 집중도 실측" 은 수치만 남고 측정
스크립트가 없었다. 여기서 그림과 숫자를 같이 만들어 재현 가능하게 한다.

마스크는 teacher(yolov8s.pt COCO) 와 GT 만으로 결정된다 — student 도 aligner 도 학습 상태도
필요 없다. 증류 지점은 head`.1` 6개(box cv2 x P3/P4/P5, cls cv3 x P3/P4/P5)뿐이고,
AMSE·QMSE 런이 둘 다 head`.1` x s-COCO 고정이라 위치 축은 하나다.

  AMSE  w_s = h*w*softmax(mean_c|t|/T),  w_c = C*softmax(mean_hw|t|/T)   ← teacher feature 만
  QMSE  cv2 지점 = 위치별 max IoU(teacher 디코딩 박스, GT),  cv3 지점 = teacher cls sigmoid max

실제 런의 AMSE 온도는 T=1.0 이다 (distiller.py `AMSELoss(temp=1.0)`). README 의 one-hot 퇴화
실측은 T=0.5 기준이라 채널 그림은 두 T 를 같이 그린다.

수식을 loss.py 에서 옮겨 적은 이상 원본이 바뀌면 그림이 조용히 틀려진다. 그래서 마지막에
`AMSELoss`·`QMSEFeatureLoss` 를 같은 입력으로 호출해 값이 일치하는지 assert 한다 (드리프트 가드).

결과: runs/detect/voc/mask_analysis/
"""

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_yolo_dataset
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.loss import AMSELoss, QMSEFeatureLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.ops import xywh2xyxy

REPO = Path(__file__).resolve().parents[2]
IMAGES = Path("C:/Users/SSAFY/datasets/VOC/images/test2007")
TEACHER = REPO / "yolov8s.pt"
OUT = REPO / "runs/detect/voc/mask_analysis"
IMGSZ = 640
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 증류 지점 6개 — distill_head1_from_8s_coco_qmse.yaml 의 layers 와 같은 순서.
# hook 은 실행 순서로 채워지는데, Detect._inference 가 boxes(cv2 전부) -> scores(cv3 전부) 순으로
# 돌아서 이 순서와 일치한다. cv2/cv3 가 스케일마다 번갈아 실행됐다면 mask_types 매핑이 어긋났을 것이다.
POINTS = [
    ("model.22.cv2.0.1", "box", "P3", "iou"),
    ("model.22.cv2.1.1", "box", "P4", "iou"),
    ("model.22.cv2.2.1", "box", "P5", "iou"),
    ("model.22.cv3.0.1", "cls", "P3", "score"),
    ("model.22.cv3.1.1", "cls", "P4", "score"),
    ("model.22.cv3.2.1", "cls", "P5", "score"),
]


def write_source_list(paths, name):
    """경로 리스트를 .txt 로 넘긴다 — 데이터셋 빌더가 파일 경로를 그대로 유지한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    txt = OUT / name
    txt.write_text("\n".join(str(Path(p).absolute()) for p in paths), encoding="utf-8")
    return str(txt)


def make_batch(stems, tag):
    """학습과 같은 전처리로 배치를 만든다.

    마스크는 letterbox 된 640 좌표계의 GT 를 쓴다. 직접 좌표 변환하면 pad/ratio 를 틀리기 쉬워서
    build_yolo_dataset(mode="val") 을 그대로 탄다 — augmentation 없이 letterbox + 정규화 xywh 가 나온다.
    """
    sources = write_source_list([IMAGES / f"{s}.jpg" for s in stems], f"_{tag}_sources.txt")
    data = check_det_dataset("VOC.yaml")
    cfg = get_cfg(DEFAULT_CFG, overrides={"imgsz": IMGSZ, "rect": False, "task": "detect", "mode": "val"})
    ds = build_yolo_dataset(cfg, sources, batch=len(stems), data=data, mode="val", rect=False)
    batch = YOLODataset.collate_fn([ds[i] for i in range(len(ds))])
    batch["img"] = batch["img"].to(DEVICE).float() / 255  # trainer.preprocess_batch 와 동일
    for k in ("bboxes", "cls", "batch_idx"):
        batch[k] = batch[k].to(DEVICE)
    # ds 는 txt 의 순서를 유지하지 않을 수 있다 — 실제 파일 순서를 돌려준다
    order = [Path(ds.im_files[i]).stem for i in range(len(ds))]
    return batch, order


def teacher_forward(model, img):
    """6지점 feature 와 eval 출력을 한 번에 받는다."""
    feats = []
    handles = [model.get_submodule(name).register_forward_hook(lambda m, i, o: feats.append(o)) for name, *_ in POINTS]
    try:
        with torch.no_grad():
            out = model(img)
    finally:
        for h in handles:
            h.remove()
    assert len(feats) == len(POINTS), f"hook {len(feats)}개 발화 — 지점 {len(POINTS)}개와 불일치"
    y = (out[0] if isinstance(out, (tuple, list)) else out).float()
    return feats, y


def amse_weights(feat, temp):
    """AMSE 의 공간·채널 가중 — loss.py AMSELoss.forward 와 같은 식."""
    t = feat.float()
    n, c, h, w = t.shape
    value = t.abs()
    w_s = (h * w * torch.softmax((value.mean(dim=1, keepdim=True) / temp).view(n, -1), dim=1)).view(n, 1, h, w)
    w_c = (c * torch.softmax(value.mean(dim=(2, 3)) / temp, dim=1)).view(n, c, 1, 1)
    return w_s, w_c


def qmse_masks(feats, y, batch):
    """지점별 QMSE 마스크 (평균 1 정규화 후) — loss.py QMSEFeatureLoss.forward 와 같은 식."""
    score_all = y[:, 4:].amax(dim=1)  # (B, A) 클래스 max
    pred_xyxy = xywh2xyxy(y[:, :4].permute(0, 2, 1))  # (B, A, 4) 픽셀 xyxy

    img_h, img_w = batch["img"].shape[-2:]
    gt = xywh2xyxy(batch["bboxes"].float()) * torch.tensor([img_w, img_h, img_w, img_h], device=y.device)
    batch_idx = batch["batch_idx"].view(-1)
    iou_all = torch.zeros_like(score_all)
    for i in range(score_all.shape[0]):
        g = gt[batch_idx == i]
        if len(g):
            iou_all[i] = bbox_iou(pred_xyxy[i].unsqueeze(1), g.unsqueeze(0), xywh=False).squeeze(-1).amax(dim=1)

    sizes = sorted({tuple(f.shape[-2:]) for f in feats}, key=lambda s: -(s[0] * s[1]))
    offsets, start = {}, 0
    for h, w in sizes:
        offsets[(h, w)] = (start, start + h * w)
        start += h * w
    assert start == score_all.shape[1], f"anchor 수 불일치: 지점 {start} vs teacher 출력 {score_all.shape[1]}"

    masks = []
    for (_, _, _, mask_type), f in zip(POINTS, feats):
        n, _, h, w = f.shape
        s0, s1 = offsets[(h, w)]
        m = (iou_all if mask_type == "iou" else score_all)[:, s0:s1].view(n, 1, h, w)
        mean = m.mean(dim=(2, 3), keepdim=True)
        masks.append(torch.where(mean > 1e-6, m / mean.clamp(min=1e-6), torch.ones_like(m)))
    return masks


def drift_guard(feats, y, batch, temp):
    """복제한 수식이 loss.py 원본과 같은 값을 내는지 확인한다.

    student feature 자리에는 teacher 에 고정 노이즈를 더한 텐서를 쓴다 — 실제 student 는 필요 없다.
    """
    gen = torch.Generator(device="cpu").manual_seed(0)
    students = [t + torch.randn(t.shape, generator=gen).to(t.device) * 0.1 for t in feats]

    # QMSE
    ref = QMSEFeatureLoss([p[3] for p in POINTS])(students, feats, batch=batch, teacher_preds=y)
    masks = qmse_masks(feats, y, batch)
    mine = sum((m * (s.float() - t.detach().float()) ** 2).mean() for m, s, t in zip(masks, students, feats))
    mine = mine / len(feats)
    assert torch.allclose(ref, mine, atol=1e-5), f"QMSE 드리프트: loss.py {ref.item():.6f} vs 스크립트 {mine.item():.6f}"

    # AMSE — 지점별로 대조한다 (KDFeatureLoss 가 지점 평균을 내기 전 단위)
    amse = AMSELoss(temp=temp)
    for (name, *_), s, t in zip(POINTS, students, feats):
        w_s, w_c = amse_weights(t, temp)
        mine_p = (w_s * w_c * (s.float() - t.float()) ** 2).mean()
        ref_p = amse(s, t)
        assert torch.allclose(ref_p, mine_p, atol=1e-5), (
            f"AMSE 드리프트 @{name}: loss.py {ref_p.item():.6f} vs 스크립트 {mine_p.item():.6f}"
        )
    print(f"  드리프트 가드 통과 — QMSE {ref.item():.4f}, AMSE 6지점 일치 (T={temp})")


def gt_cell_grid(boxes_xyxy, h, w, img_size):
    """셀 중심이 GT 박스 안에 들어가는지 (h,w) bool 격자."""
    ys = (np.arange(h) + 0.5) * (img_size / h)
    xs = (np.arange(w) + 0.5) * (img_size / w)
    gx, gy = np.meshgrid(xs, ys)
    inside = np.zeros((h, w), dtype=bool)
    for x1, y1, x2, y2 in boxes_xyxy:
        inside |= (gx >= x1) & (gx <= x2) & (gy >= y1) & (gy <= y2)
    return inside


def mask_stats(mask, inside):
    """정규화된 마스크의 집중도 — README §4 문구와 같은 정의."""
    m = mask.reshape(-1).astype(np.float64)
    n = m.size
    total = m.sum()
    if total <= 0:
        return dict(participation=1.0, top1pct_mass=0.01, frac_below_0_1=0.0, gt_mass=0.0)
    p = m / total
    nz = p[p > 0]
    participation = math.exp(-(nz * np.log(nz)).sum()) / n  # 유효 참여율 (엔트로피 기준)
    k = max(1, int(math.ceil(n * 0.01)))
    top1 = np.sort(m)[::-1][:k].sum() / total
    return dict(
        participation=participation,
        top1pct_mass=top1,
        frac_below_0_1=float((m < 0.1).mean()),
        gt_mass=float(m[inside.reshape(-1)].sum() / total),
    )


def overlay(ax, img_rgb, mask, gt_xyxy, title, vmax):
    """마스크를 이미지 위에 얹는다. mask=None 이면 원본만.

    알파를 마스크 값에 비례시킨다 — 균일 알파로 덮으면 가중이 0 인 배경까지 컬러맵 바닥색으로
    칠해져 원본이 죽는다. vmax 는 99.5 분위로 자른다: score 마스크는 최댓값이 수백이라
    max 로 정규화하면 소수 픽셀 빼고 전부 검게 깔린다 (제목에는 실제 max 를 적는다).
    """
    ax.imshow(img_rgb)
    if mask is not None:
        up = np.kron(mask, np.ones((img_rgb.shape[0] // mask.shape[0], img_rgb.shape[1] // mask.shape[1])))
        hi = vmax or max(np.percentile(up, 99.5), 1e-6)
        ax.imshow(up, cmap="inferno", alpha=np.clip(up / hi, 0, 1) * 0.85, vmin=0, vmax=hi)
    for x1, y1, x2, y2 in gt_xyxy:
        ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="white", linewidth=1.2))
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def render(stem, img_rgb, gt_xyxy, panels, path, suptitle):
    """panels: [(row, col, mask|None, title)] 로 2x3 격자를 그린다."""
    rows = max(p[0] for p in panels) + 1
    cols = max(p[1] for p in panels) + 1
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows))
    axes = np.atleast_2d(axes)
    for r, c, mask, title, vmax in panels:
        overlay(axes[r][c], img_rgb, mask, gt_xyxy, title, vmax)
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def channel_figure(feats, temps, path):
    """AMSE 채널 가중 w_c — T 별로 정렬해 겹쳐 그린다 (one-hot 퇴화 확인용)."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for idx, (name, branch, scale, _) in enumerate(POINTS):
        ax = axes[idx // 3][idx % 3]
        for temp in temps:
            _, w_c = amse_weights(feats[idx][:1], temp)
            v = np.sort(w_c.view(-1).cpu().numpy())[::-1]
            ax.plot(v, label=f"T={temp}  max={v[0]:.1f}/{v.size}")
        ax.set_yscale("log")
        ax.set_title(f"{branch} {scale}  ({name})", fontsize=9)
        ax.set_xlabel("channel (sorted)")
        ax.set_ylabel("w_c")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("AMSE channel weight — mean is exactly 1, so max/C shows the collapse", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def collect_stats(model, stems, temp):
    """표본 이미지에 대해 지점 x 마스크별 집중도를 모은다."""
    batch, order = make_batch(stems, "stats")
    feats, y = teacher_forward(model, batch["img"])
    qmasks = qmse_masks(feats, y, batch)
    img_h = batch["img"].shape[-1]
    gt = xywh2xyxy(batch["bboxes"].float()) * torch.tensor([img_h, img_h, img_h, img_h], device=y.device)
    batch_idx = batch["batch_idx"].view(-1)

    rows = []
    for i, stem in enumerate(order):
        g = gt[batch_idx == i].cpu().numpy()
        for idx, (name, branch, scale, mask_type) in enumerate(POINTS):
            _, _, h, w = feats[idx].shape
            inside = gt_cell_grid(g, h, w, img_h)
            w_s, w_c = amse_weights(feats[idx][i : i + 1], temp)
            for label, m in (
                (f"amse_ws(T={temp})", w_s[0, 0].cpu().numpy()),
                (f"qmse_{mask_type}", qmasks[idx][i, 0].cpu().numpy()),
            ):
                s = mask_stats(m, inside)
                rows.append(
                    dict(
                        image=stem, point=name, branch=branch, scale=scale, mask=label,
                        cells=h * w, gt_area_frac=float(inside.mean()), mask_max=float(m.max()),
                        wc_max=float(w_c.max()) if label.startswith("amse") else "",
                        wc_max_ratio=float(w_c.max() / w_c.numel()) if label.startswith("amse") else "",
                        **{k: round(v, 6) for k, v in s.items()},
                    )
                )
    return rows


def write_stats(rows, temp):
    """원자료 CSV 와 README 대조용 범위 요약."""
    with open(OUT / "mask_stats.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n_img = len({r["image"] for r in rows})
    lines = [
        f"표본 {n_img}장, 정규화 후. 값은 이미지 평균 [최소~최대].",
        "README §4 의 범위는 지점(스케일)별 평균들이 만드는 폭과 비교한다.",
        "",
        f"{'mask':22s} {'branch/scale':13s} {'참여율':>20s} {'상위1%질량':>20s} {'w<0.1':>20s} {'GT내질량':>20s}",
    ]
    for mask in sorted({r["mask"] for r in rows}):
        for branch in ("box", "cls"):
            for scale in ("P3", "P4", "P5"):
                sel = [r for r in rows if r["mask"] == mask and r["branch"] == branch and r["scale"] == scale]
                if not sel:
                    continue

                def cell(k, sel=sel):
                    v = [r[k] for r in sel]
                    return f"{sum(v) / len(v) * 100:5.1f} [{min(v) * 100:4.1f}~{max(v) * 100:5.1f}]"

                lines.append(
                    f"{mask:22s} {branch + ' ' + scale:13s} {cell('participation'):>20s} "
                    f"{cell('top1pct_mass'):>20s} {cell('frac_below_0_1'):>20s} {cell('gt_mass'):>20s}"
                )
    gt_area = [r["gt_area_frac"] for r in rows]
    lines += [
        "",
        f"균등 마스크 기대치(GT 영역 면적 비율): 평균 {sum(gt_area) / len(gt_area) * 100:.1f}% "
        f"[{min(gt_area) * 100:.1f}~{max(gt_area) * 100:.1f}%]",
    ]
    wc = [(r["wc_max"], r["wc_max_ratio"], r["branch"], r["scale"]) for r in rows if r["wc_max"] != ""]
    if wc:
        top = max(wc)
        lines.append(f"AMSE 채널 가중 max (T={temp}): {top[0]:.1f} ({top[1] * 100:.1f}% of C) @ {top[2]} {top[3]}")
    (OUT / "mask_stats_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", nargs="+", default=["003076", "009602"], help="렌더링할 test2007 stem")
    p.add_argument("--stats-images", type=int, default=16, help="통계 표본 수 (README 실측이 16장)")
    p.add_argument("--temp", type=float, default=1.0, help="AMSE 온도 — 실제 런은 1.0")
    p.add_argument("--scale", default="P3", help="compare 그림에 쓸 스케일")
    p.add_argument("--vmax", type=float, default=0, help="0=패널별 자동, >0=공유 스케일")
    args = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    model = YOLO(TEACHER).model.to(DEVICE).eval()

    # --- 렌더링 ---
    print(f"[1/3] rendering {len(args.images)} images")
    batch, order = make_batch(args.images, "render")
    feats, y = teacher_forward(model, batch["img"])
    qmasks = qmse_masks(feats, y, batch)
    drift_guard(feats, y, batch, args.temp)

    img_h = batch["img"].shape[-1]
    gt = xywh2xyxy(batch["bboxes"].float()) * torch.tensor([img_h, img_h, img_h, img_h], device=y.device)
    batch_idx = batch["batch_idx"].view(-1)

    for i, stem in enumerate(order):
        img_rgb = (batch["img"][i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        g = gt[batch_idx == i].cpu().numpy()

        amse_panels, qmse_panels = [], []
        for idx, (_, branch, scale, mask_type) in enumerate(POINTS):
            r, c = (0 if branch == "box" else 1), ("P3", "P4", "P5").index(scale)
            w_s, _ = amse_weights(feats[idx][i : i + 1], args.temp)
            ws = w_s[0, 0].cpu().numpy()
            qm = qmasks[idx][i, 0].cpu().numpy()
            amse_panels.append((r, c, ws, f"AMSE w_s  {branch} {scale}  max={ws.max():.1f}", args.vmax))
            qmse_panels.append((r, c, qm, f"QMSE {mask_type}  {branch} {scale}  max={qm.max():.1f}", args.vmax))

        render(stem, img_rgb, g, amse_panels, OUT / f"{stem}_amse.jpg",
               f"{stem} — AMSE spatial weight (activation magnitude, T={args.temp})")
        render(stem, img_rgb, g, qmse_panels, OUT / f"{stem}_qmse.jpg",
               f"{stem} — QMSE quality mask (IoU on box branch, score on cls branch)")

        # 장표용: 스케일 하나 고정, 분기별로 원본 / AMSE / QMSE 3열
        sel = args.scale
        cmp_panels = []
        for row, branch in enumerate(("box", "cls")):
            idx = next(j for j, pt in enumerate(POINTS) if pt[1] == branch and pt[2] == sel)
            w_s, _ = amse_weights(feats[idx][i : i + 1], args.temp)
            ws = w_s[0, 0].cpu().numpy()
            qm = qmasks[idx][i, 0].cpu().numpy()
            cmp_panels += [
                (row, 0, None, f"{branch} branch {sel} — image + GT", 0),
                (row, 1, ws, f"AMSE w_s (magnitude)  max={ws.max():.1f}", args.vmax),
                (row, 2, qm, f"QMSE {POINTS[idx][3]} (quality)  max={qm.max():.1f}", args.vmax),
            ]
        render(stem, img_rgb, g, cmp_panels, OUT / f"{stem}_compare.jpg",
               f"{stem} — weighting criterion: magnitude (AMSE) vs prediction quality (QMSE), {sel}")

    # --- 채널 가중 ---
    print("[2/3] channel weights (T=0.5 vs 1.0)")
    channel_figure(feats, [0.5, args.temp], OUT / "amse_channel_weights.png")

    # --- 통계 ---
    stems = sorted(f.stem for f in IMAGES.glob("*.jpg"))[: args.stats_images]
    print(f"[3/3] stats over {len(stems)} images")
    write_stats(collect_stats(model, stems, args.temp), args.temp)
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
