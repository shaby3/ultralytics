"""best.pt vs best_converted.pt — GC10-DET test split 비교 검증.

best.pt는 C2f 아키텍처 변경(cv1 단일 → cv0/cv1 분리) 이전에 학습된 가중치로,
현재 코드에서는 shape mismatch 또는 AttributeError가 발생할 수 있음.
best_converted.pt는 convert_weights.py로 변환 완료된 가중치.
"""

from ultralytics import YOLO

WEIGHTS = {
    "best": "runs/detect/gc10_det/train/baseline_yolo26n_gc10_det_200epoch/weights/best.pt",
    "best_converted": "runs/detect/gc10_det/train/baseline_yolo26n_gc10_det_200epoch/weights/best_converted.pt",
}

VAL_KWARGS = dict(
    data="GC10-DET.yaml",
    split="test",
    imgsz=640,
    batch=32,
    workers=2,
    project="runs/detect/gc10_det/val_compare",
    exist_ok=True,
    device=0,
)

if __name__ == "__main__":
    results = {}
    errors = {}

    for tag, weight in WEIGHTS.items():
        print(f"\n{'='*60}")
        print(f"  Validating: {tag}  ({weight})")
        print(f"{'='*60}")
        try:
            metrics = YOLO(weight).val(**VAL_KWARGS, name=tag)
            results[tag] = metrics
        except Exception as e:
            errors[tag] = e
            print(f"[SKIP] {tag} 로드/검증 실패: {e}")
            print("  → C2f 구조 변경(cv0/cv1 분리)으로 인한 호환성 문제일 수 있습니다.")
            print("  → convert_weights.py 로 변환 후 재시도하세요.")

    print("\n" + "=" * 60)
    print(f"{'Model':<20} {'mAP50':>10} {'mAP50-95':>12}  {'비고':>10}")
    print("-" * 60)
    for tag in WEIGHTS:
        if tag in results:
            m = results[tag]
            print(f"{tag:<20} {m.box.map50:>10.4f} {m.box.map:>12.4f}")
        else:
            print(f"{tag:<20} {'--':>10} {'--':>12}  (오류: {type(errors[tag]).__name__})")
    print("=" * 60)
