# trackers/ - 객체 트래킹

## 개요
YOLO의 predict 모드와 통합되는 다중 객체 트래킹 알고리즘. ByteTrack과 BoT-SORT 알고리즘 지원.

## 핵심 파일
- `track.py` — 진입점: `on_predict_start()`로 트래커 초기화, `on_predict_postprocess_end()`로 트랙 업데이트. `TRACKER_MAP = {"bytetrack": BYTETracker, "botsort": BOTSORT}`. 트래커 설정은 `cfg/trackers/` YAML에서 로드.
- `basetrack.py` — BaseTrack 기본 클래스, TrackState 열거형 (New, Tracked, Lost, Removed)
- `byte_tracker.py` — BYTETracker: ByteTrack 알고리즘 구현 (STrack 사용)
- `bot_sort.py` — BOTSORT: BoT-SORT 알고리즘 (ByteTrack 확장 + ReID + GMC)

## 패턴
- 트래커는 콜백으로 예측에 훅 (on_predict_start, on_predict_postprocess_end)
- 각 트래커는 칼만 필터 예측으로 트랙 상태 유지
- 트래커 설정은 `cfg/trackers/` (botsort.yaml, bytetrack.yaml)에 위치

## 하위 디렉토리
- `utils/` — 트래킹 유틸리티: kalman_filter.py (칼만 필터), matching.py (IoU/임베딩 매칭), gmc.py (Global Motion Compensation)
