# hub/ - Ultralytics HUB 통합

## 개요
Ultralytics HUB 클라우드 플랫폼과의 클라이언트 측 통합. 모델 학습, 관리, 배포를 위한 로그인/로그아웃, 모델 업로드/내보내기, 학습 세션 관리 제공.

## 핵심 파일
- `__init__.py` — 공개 API: `login()`, `logout()`, `reset_model()`, `export_model()`, `check_dataset()`, `export_fmts_hub()`
- `auth.py` — Auth 클래스: API 키 인증
- `session.py` — HUBTrainingSession: 클라우드 연결 학습 (하트비트, 모델 업로드)
- `utils.py` — HUB_API_ROOT, HUB_WEB_ROOT, PREFIX, 요청 헬퍼

## 패턴
- hub-sdk 패키지 (외부)를 사용하여 API 통신
- 인증은 SETTINGS (`utils/__init__.py`)에 api_key로 저장
- HUBTrainingSession이 학습 콜백에 훅하여 진행 상황 보고

## 하위 디렉토리
- `google/` — Google Cloud 통합 (Colab 지원)
