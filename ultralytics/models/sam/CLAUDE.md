# models/sam/ - Segment Anything Model

## 개요
SAM (Segment Anything Model) 및 SAM3 구현. 포인트, 박스, 마스크 프롬프트를 사용한 이미지 세그멘테이션 지원.

## 핵심 파일
- `model.py` — SAM, SAM3 모델 클래스
- `build.py` — SAM 모델 빌더 함수 (sam_b, sam_l, sam_h)
- `build_sam3.py` — SAM3 변형 빌더 함수
- `predict.py` — SAM 전용 프레딕터 (프롬프트 기반 추론)
- `amg.py` — Automatic Mask Generation 유틸리티

## 하위 디렉토리
- `modules/` — SAM 신경망 컴포넌트: encoders.py (이미지 인코더), decoders.py (마스크 디코더), transformer.py (양방향 트랜스포머), blocks.py, sam.py, memory_attention.py, tiny_encoder.py
- `sam3/` — SAM3 아키텍처: encoder.py, decoder.py, vitdet.py, necks.py, text_encoder_ve.py, geometry_encoders.py, maskformer_segmentation.py
