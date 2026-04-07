# solutions/ - 사전 구축 CV 솔루션

## 개요
YOLO 위에 구축된 즉시 사용 가능한 컴퓨터 비전 솔루션. 각 솔루션은 BaseSolution을 상속하며, 탐지/트래킹과 도메인별 로직(카운팅, 속도, 히트맵 등)을 결합.

## 핵심 파일
- `solutions.py` — BaseSolution 기본 클래스: YOLO 모델 로드, 트래킹 관리, 영역 초기화, 어노테이션
- `config.py` — SolutionConfig: 솔루션별 설정
- `object_counter.py` — ObjectCounter: 라인/영역 통과 객체 카운팅
- `region_counter.py` — RegionCounter: 정의 영역 내 객체 카운팅
- `speed_estimation.py` — SpeedEstimator: 객체 속도 추정
- `distance_calculation.py` — DistanceCalculation: 객체 간 거리 측정
- `heatmap.py` — Heatmap: 활동 히트맵 생성
- `analytics.py` — Analytics: 시각화 분석
- `ai_gym.py` — AIGym: 운동 카운팅
- `parking_management.py` — ParkingManagement: 주차장 모니터링
- `queue_management.py` — QueueManager: 대기열 모니터링
- `security_alarm.py` — SecurityAlarm: 침입 감지
- `streamlit_inference.py` — Streamlit 웹 UI 추론

## 패턴
- 모든 솔루션은 내부적으로 YOLO 트래킹 모드 사용
- 솔루션은 영역/라인 정의를 설정으로 수신
- BaseSolution이 모델 로딩, 트래킹, 어노테이션 셋업 처리

## 하위 디렉토리
- `templates/` — 웹 기반 솔루션용 HTML 템플릿
