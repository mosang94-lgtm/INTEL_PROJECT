# 소스코드 구조 및 동작 방식

## 프로젝트 개요

차량 이미지에서 **손상 부위(damage)** 및 **차량 부품(part)** 을 픽셀 단위로 분류하는  
**의미론적 분할(Semantic Segmentation)** AI 모델.  
U-Net (ResNet34 encoder) 아키텍처를 사용하며, COCO 포맷 어노테이션 기반으로 학습/평가한다.

---

## 디렉토리 및 파일 구조

```
workspace/
├── 1.AI모델 소스코드/
│   └── code/                          ← 실제 실행 코드 루트
│       ├── main.py                    ← 진입점 (학습/평가 실행)
│       ├── requirements.txt           ← 패키지 의존성
│       ├── Dockerfile                 ← Docker 환경 설정
│       ├── damage_labeling.csv        ← 손상 데이터 분할 정보 (train/val/test)
│       ├── part_labeling.csv          ← 부품 데이터 분할 정보 (train/val/test)
│       └── src/
│           ├── Models.py              ← U-Net 모델 정의
│           ├── Datasets.py            ← PyTorch Dataset (COCO 포맷 로더)
│           ├── Train.py               ← 학습 루프 (Trainer 클래스)
│           ├── Evaluation.py          ← 평가 루프 (Evaluation 클래스)
│           └── Utils.py               ← 손실함수, 평가지표, 데이터 전처리 유틸
│
└── New_Sample/
    └── 라벨링데이터/
        └── TL_damage/damage/          ← 원본 어노테이션 JSON 샘플
```

### 실행 시 생성되는 디렉토리 구조 (data/)

```
data/                                  ← code/ 기준 ../data/
├── Dataset/
│   ├── 1.원천데이터/
│   │   ├── damage/                    ← 손상 이미지 (.jpg)
│   │   └── damage_part/               ← 부품 이미지 (.jpg)
│   └── 2.라벨링데이터/
│       ├── damage/                    ← 손상 어노테이션 (.json)
│       └── damage_part/               ← 부품 어노테이션 (.json)
├── datainfo/                          ← COCO 포맷 변환된 JSON
│   ├── damage_Scratched_train.json
│   ├── damage_Separated_train.json
│   ├── damage_Crushed_train.json
│   ├── damage_Breakage_train.json
│   ├── damage_val.json
│   ├── damage_test.json
│   ├── part_train.json
│   ├── part_val.json
│   └── part_test.json
├── weight/                            ← 저장된 모델 가중치 (.pt)
└── result_log/                        ← 학습/평가 결과 로그 (.json)
```

---

## 각 파일 역할

### `main.py` — 진입점
CLI 인자를 파싱하여 학습 또는 평가 흐름을 제어한다.

```
python main.py --train train --task damage --label all
python main.py --eval y    --task damage --dataset val
python main.py --train train --task part --cls 16
python main.py --eval y    --task part   --dataset test --weight_file [PART]Unet.pt
```

| 인자 | 설명 |
|---|---|
| `--train` | 학습 실행 (값: `train`) |
| `--eval` | 평가 실행 (값: `y`) |
| `--task` | 작업 유형 (`damage` / `part`) |
| `--method` | 모델 방식 (`multi` / `single`) |
| `--label` | 학습할 손상 라벨 (`all` / `2` / `3` 등) |
| `--cls` | 분류 클래스 수 (part: 16) |
| `--dataset` | 평가 데이터셋 (`val` / `test`) |
| `--weight_file` | 불러올 가중치 파일명 |

---

### `src/Models.py` — 모델 정의
`segmentation_models_pytorch`의 `Unet`을 래핑한 클래스.

```
Unet
├── encoder: ResNet34 (ImageNet 사전학습)
├── decoder: U-Net 디코더
└── head:    클래스 수만큼 출력 채널 (damage: 2, part: 16)
```

---

### `src/Datasets.py` — 데이터 로더
COCO 포맷 JSON을 읽어 이미지와 마스크를 반환하는 PyTorch `Dataset`.

```
입력: COCO JSON 경로, 이미지 기본 경로
처리:
  ① COCO API로 이미지 ID 목록 로드
  ② 이미지 파일 읽기 (cv2) → BGR→RGB 변환
  ③ 어노테이션으로 세그멘테이션 마스크 생성
     - damage (one_channel=True): 해당 카테고리만 이진 마스크
     - part   (one_channel=False): 카테고리 ID를 픽셀값으로 하는 다중 마스크
  ④ albumentations transform 적용 (리사이즈, 증강)
  ⑤ 정규화 (÷255), HWC→CHW 전치
출력: (images, masks, file_name) 튜플
```

---

### `src/Train.py` — 학습 클래스 `Trainer`

```
Trainer.__init__()
  ├── Datasets 로드 (train / val)
  ├── Adam 옵티마이저 설정 (encoder_lr / decoder_lr 분리)
  └── 학습 로그 구조 초기화

Trainer.train()
  └── for epoch in range(epochs):
        ├── [학습] DataLoader → forward → loss → backward → step
        ├── [100 step마다] loss 출력
        ├── [lr_scheduler] StepLR step
        └── validation() 호출
              ├── argmax → IoU 계산 (per image)
              ├── 전체 mIoU 계산
              └── best mIoU 갱신 시 모델 저장 (.pt)

로그 저장 위치: ../data/result_log/[task]train_log.json
모델 저장 위치: ../data/weight/Unet_...epoch_IoU_....pt
```

---

### `src/Evaluation.py` — 평가 클래스 `Evaluation`

```
Evaluation.evaluation()
  ├── [damage task] 4개 모델 순차 로드 (Scratched / Separated / Crushed / Breakage)
  │     └── 각 모델별 validation() 실행
  └── [part task] 단일 모델 로드
        └── validation() 실행

validation()
  ├── tqdm으로 배치 순회
  ├── forward → loss 계산
  ├── argmax → confusion matrix 누적
  └── 최종 mIoU, per-class IoU 계산 및 로그 저장

로그 저장 위치: ../data/result_log/[task]_val/test_evaluation_log.json
```

---

### `src/Utils.py` — 유틸리티

| 항목 | 설명 |
|---|---|
| `RemakeCOCOformat` | 원본 어노테이션 JSON을 COCO 표준 포맷으로 변환. `__main__` 실행 시 사용 |
| `label_split()` | 라벨별 샘플 파일을 무작위 추출하여 개별 COCO JSON 생성 |
| `label_accuracy_score(hist)` | Confusion Matrix에서 acc, mIoU, fwavacc, per-class IoU 계산 |
| `add_hist()` / `_fast_hist()` | 배치 단위 confusion matrix 누적 |
| `FocalLoss` | Focal Loss 구현 (현재 main.py에서는 CrossEntropyLoss를 사용 중) |

---

## 전체 데이터 흐름

```
[원본 데이터]
  이미지 (.jpg)  +  어노테이션 (.json, 자체 포맷)
         │
         ▼
[전처리] python src/Utils.py --make_cocoformat --task all
  RemakeCOCOformat → COCO 표준 JSON 생성
  (data/datainfo/*.json)
         │
         ▼
[학습] python main.py --train train --task damage --label all
  main.py
    └── Trainer (Train.py)
          ├── Datasets (Datasets.py) ──▶ 이미지/마스크 배치
          ├── Unet (Models.py)       ──▶ 예측 마스크
          ├── CrossEntropyLoss       ──▶ 손실 계산
          └── Adam optimizer         ──▶ 가중치 업데이트
                                         │
                                         ▼
                                    weight/*.pt 저장
         │
         ▼
[평가] python main.py --eval y --task damage --dataset val
  main.py
    └── Evaluation (Evaluation.py)
          ├── Datasets (Datasets.py) ──▶ 이미지/마스크 배치
          ├── 저장된 weight 로드     ──▶ 추론
          ├── label_accuracy_score   ──▶ mIoU, per-class IoU
          └── result_log/*.json      ──▶ 결과 저장
```

---

## Task별 모델 구조 비교

| 구분 | damage | part |
|---|---|---|
| 출력 클래스 수 | 2 (배경 / 손상) | 16 (배경 + 15개 부품) |
| 모델 수 | 4개 (손상 유형별 별도 학습) | 1개 |
| 손상 유형 | Scratched / Separated / Crushed / Breakage | - |
| one_channel | True | False |
| 평가 방식 | 4개 모델 순차 평가 후 종합 | 단일 모델 평가 |

---

## 사용된 주요 외부 라이브러리

| 라이브러리 | 역할 |
|---|---|
| `torch` | 딥러닝 프레임워크 |
| `segmentation_models_pytorch` | U-Net + ResNet34 구현 제공 |
| `albumentations` | 이미지 증강 (Resize, RandomRotate90, CoarseDropout) |
| `pycocotools` | COCO 포맷 어노테이션 파싱 |
| `opencv-python` | 이미지 읽기/색공간 변환 |
| `pandas` | CSV 기반 데이터셋 분할 정보 읽기 |
| `numpy` | 배열 연산, confusion matrix 계산 |
| `pytz` | 한국 시간대(KST) 학습 로그 기록 |
| `tqdm` | 평가 루프 진행률 표시 |
