# nn/modules/ - 신경망 빌딩 블록

## 개요
YOLO 모델 구성에 사용되는 모든 재사용 가능한 신경망 레이어/블록. YAML 설정의 모듈 이름이 여기 정의된 클래스로 매핑됨.

## 핵심 파일
- `block.py` — 핵심 블록: C3, C3k2, C2f, C2PSA, SPPF, ADown, SCDown, HGBlock, HGStem, RepNCSPELAN4, ELAN1, PSA, A2C2f, Bottleneck, Proto, DFL 등
- `conv.py` — 컨볼루션 레이어: Conv (Conv2d+BN+SiLU), DWConv, GhostConv, RepConv, Conv2, ConvTranspose, DWConvTranspose2d, Focus, ChannelAttention, SpatialAttention, CBAM
- `head.py` — 태스크 헤드: Detect, Segment, Pose, OBB, Classify, RTDETRDecoder, WorldDetect, YOLOEDetect, YOLOESegment, v10Detect, LRPCHead, v26 변형 (OBB26, Pose26, Segment26, YOLOESegment26)
- `transformer.py` — 트랜스포머 컴포넌트: AIFI, TransformerBlock, TransformerLayer, MLPBlock, LayerNorm2d, DeformableTransformerDecoder, MSDeformAttn
- `activation.py` — 커스텀 활성화 함수: AGLU, Mish
- `utils.py` — 유틸리티: bias_init, multi_scale_deformable_attn 등

## 패턴
- 모든 모듈은 `torch.nn.Module` 규약을 따름
- 컨볼루션 블록은 기본적으로 BatchNorm + SiLU 활성화 포함
- YAML 설정의 모듈 이름은 여기 클래스 이름과 정확히 일치해야 함
- `__init__.py`가 모든 클래스를 re-export하여 flat import 가능
