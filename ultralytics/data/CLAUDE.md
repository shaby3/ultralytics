# data/ - 데이터 로딩 및 증강

## 개요
데이터셋 로딩, 증강 파이프라인, 포맷 변환, 데이터로더 구축을 처리.

## 핵심 파일
- `build.py` — InfiniteDataLoader, `build_dataloader()`, `load_inference_source()`. 다양한 소스 타입 (이미지, 비디오, 스트림, 텐서) 지원 데이터로더 구축.
- `dataset.py` — YOLODataset, YOLOMultiModalDataset, GroundingDataset. BaseDataset을 상속하는 핵심 데이터셋 클래스.
- `base.py` — BaseDataset 클래스: 캐싱, 라벨 로딩, 트랜스폼.
- `augment.py` — 증강 파이프라인: BaseTransform, Mosaic, MixUp, CutMix, CopyPaste, RandomPerspective, LetterBox, Albumentations 래퍼.
- `loaders.py` — 추론 소스 로더: LoadImagesAndVideos, LoadScreenshots, LoadStreams, LoadPilAndNumpy, LoadTensor.
- `converter.py` — 데이터셋 포맷 변환기 (COCO, DOTA 등)
- `annotator.py` — 자동 어노테이션 유틸리티
- `utils.py` — 헬퍼 함수: check_det_dataset, check_cls_dataset, IMG_FORMATS, VID_FORMATS

## 패턴
- 증강은 `__call__` 체이닝으로 합성
- 데이터셋 클래스는 첫 로드 시 라벨을 디스크에 캐싱
- 학습은 InfiniteDataLoader를 사용하여 에폭 전환 시 끊김 없음

## 하위 디렉토리
- `scripts/` — 데이터셋 다운로드 셸 스크립트 (get_coco.sh, get_imagenet.sh 등)
