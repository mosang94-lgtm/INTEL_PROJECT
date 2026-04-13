# 하이퍼파라미터 튜닝 보고서

## 프로젝트 목표
- Accuracy: 95% 이상
- F1 Score: 0.9 이상

---

## 1. Baseline (Before) 성능

### 학습 설정

| 항목 | 설정값 |
|---|---|
| 모델 | U-Net (ResNet34 encoder, ImageNet pretrained) |
| 입력 크기 | 256 x 256 |
| Batch Size | 64 |
| Loss | CrossEntropyLoss |
| Optimizer | Adam |
| Encoder LR | 1e-6 |
| Decoder LR | 3e-4 |
| Weight Decay | 0 |
| LR Scheduler | 없음 |
| Augmentation | 없음 (Resize만 적용) |
| Best 모델 기준 | mIoU |
| Early Stopping | 없음 (100 epoch 고정) |

### ResNet34 Encoder 아키텍처

| 파라미터 | 값 |
|---|---|
| Block 타입 | BasicBlock (3x3 conv x 2) |
| Expansion | 1 |
| Stride | Layer1: 1, Layer2~4: 2 |
| Layer 구성 | [3, 4, 6, 3] |
| 파라미터 수 | 21.8M |
| 최종 채널 | 512 |

### Scratched (label 0) — Epoch 19 Best 모델 평가 결과 (val set)

| 지표 | 값 | 목표 | 달성 |
|---|---|---|---|
| **Accuracy** | 96.49% | 95%+ | O |
| **F1 Score** | 0.6540 | 0.9+ | X |
| Precision | 0.7227 | - | - |
| Recall | 0.5972 | - | - |
| mIoU | 0.7103 | - | - |
| Background IoU | 0.9627 | - | - |
| Target IoU | 0.4578 | - | - |

### Epoch별 학습 추이 (Scratched Baseline)

| Epoch | Val Loss | mIoU | Background IoU | Target IoU |
|---|---|---|---|---|
| 1 | 0.1192 | 0.6733 | 0.9555 | 0.3910 |
| 4 | 0.0956 | 0.6931 | 0.9607 | 0.4254 |
| 8 | 0.0964 | 0.7045 | 0.9588 | 0.4502 |
| 12 | 0.0953 | 0.7083 | 0.9614 | 0.4553 |
| 19 (Best) | 0.0993 | **0.7103** | 0.9627 | **0.4578** |
| 22 | 0.0997 | 0.7084 | 0.9629 | 0.4539 |

### Baseline 문제점 분석

1. **클래스 불균형 미대응**: 배경 픽셀이 전체의 95% 이상 → CrossEntropyLoss가 배경 위주로 최적화되어 손상 영역 검출력 부족
2. **낮은 Recall (0.60)**: 실제 손상의 40%를 놓치고 있음 (과소 검출)
3. **Augmentation 미적용**: 학습 데이터 다양성 부족으로 과적합 위험, 특히 스크래치 같은 가는 형태에 취약
4. **LR Scheduler 미적용**: 학습 후반부에 학습률 조정이 없어 미세 수렴 불가
5. **Encoder 학습 부족**: Encoder LR 1e-6이 너무 낮아 도메인 특화 feature 학습 제한적
6. **낮은 해상도 (256x256)**: 스크래치 같은 가느다란 손상이 축소되면서 디테일 소실
7. **가벼운 Encoder (ResNet34)**: BasicBlock 기반으로 미세한 feature 추출에 한계

---

## 2. 1차 튜닝 (Loss / Augmentation / LR Scheduler / Encoder LR)

### 변경 내용

#### 튜닝 1: Loss 함수 변경

| Before | After |
|---|---|
| `CrossEntropyLoss()` | `DiceBCELoss(bce_weight=0.5, smooth=1.0)` |

**변경 이유**
- Dice Loss는 F1 Score와 수학적으로 동일한 구조 (2*intersection / union)
- 클래스 불균형 상황에서 소수 클래스의 intersection을 직접 최적화
- BCE와 조합하여 학습 초기 안정성 확보
- `DiceLoss = 1 - (2*TP) / (2*TP + FP + FN)` → Recall과 Precision 균형 개선

#### 튜닝 2: 데이터 증강 (Augmentation) 추가

| Before | After |
|---|---|
| Resize(256) 만 적용 | 5종 증강 + Resize(256) |

**적용된 증강 기법:**

| 기법 | 확률 | 효과 |
|---|---|---|
| RandomRotate90 | p=0.3 | 회전 불변성 학습, 방향성 손상 대응 |
| HorizontalFlip | p=0.5 | 좌우 대칭 데이터 2배 효과 |
| VerticalFlip | p=0.2 | 상하 대칭 다양성 확보 |
| ColorJitter | p=0.3 | 조명/색상 변화 강건성 향상 |
| CoarseDropout | p=0.2 | 부분 가림(occlusion)에 대한 강건성 |

**변경 이유**: part 학습에는 이미 Augmentation이 적용되어 있었으나, damage 학습에는 빠져 있었음. 스크래치 같은 가늘고 긴 형태는 회전/뒤집기 증강이 특히 효과적.

#### 튜닝 3: LR Scheduler 적용

| Before | After |
|---|---|
| 없음 (고정 LR) | `CosineAnnealingLR(T_max=100, eta_min=1e-7)` |

**변경 이유**
- 학습 초반: 큰 LR로 빠르게 수렴
- 학습 후반: 작은 LR로 미세 조정 (fine-tuning)
- Cosine 스케줄이 세그멘테이션 태스크에서 StepLR 대비 일반적으로 성능 우수

#### 튜닝 4: Encoder Learning Rate 상향

| Before | After |
|---|---|
| Encoder LR: 1e-6 | Encoder LR: **1e-5** (10배 상향) |

**변경 이유**: Encoder LR이 너무 낮으면 ImageNet 가중치에서 벗어나지 못해 차량 손상 도메인에 특화된 feature 학습이 부족. 10배 상향으로 fine-tuning 강도를 높임.

#### 기타 변경

| 항목 | Before | After | 이유 |
|---|---|---|---|
| Weight Decay | 0 | **1e-4** | 과적합 억제 |
| Best 모델 기준 | mIoU | **F1 Score** | 프로젝트 목표 지표에 직접 맞춤 |
| Early Stopping | 없음 | **patience=15** | 불필요한 학습 시간 절약 |

### 1차 튜닝 결과 (val set)

| 모델 | Best F1 | Accuracy | mIoU | Best Epoch |
|---|---|---|---|---|
| **Scratched** | 0.6769 | 96.49% | 0.7376 | 7 |
| **Separated** | 0.4958 | 98.83% | 0.6589 | 15 |
| **Crushed** | 0.4663 | 97.93% | 0.6416 | 15 |
| **Breakage** | 0.5287 | 98.78% | 0.6735 | 18 |

### 1차 튜닝 분석

- Scratched F1: 0.6540 → **0.6769** (+0.023 개선) — Baseline 대비 소폭 상승
- DiceBCELoss 효과로 초기 수렴이 빨라짐 (epoch 2에서 이미 baseline 근접)
- **그러나 F1 0.9 목표에는 여전히 크게 부족**

### 1차 튜닝 한계 원인

1. **해상도 256x256의 근본적 한계**: 스크래치 같은 가느다란 손상은 256x256에서 디테일이 소실되어 픽셀 단위 F1이 구조적으로 낮을 수밖에 없음
2. **ResNet34 Encoder의 feature 추출력 부족**: BasicBlock 기반(expansion=1)으로 최종 채널이 512에 그쳐 미세한 손상 패턴 캡처에 한계
3. **경계 불확실성**: 라벨링 자체의 픽셀 단위 오차 → 모델이 맞춰도 오답 처리되는 경우 존재

---

## 3. 2차 튜닝 (Encoder 변경 + 해상도 증가)

1차 튜닝의 한계를 해결하기 위해 모델 구조와 입력 해상도를 변경.

### 변경 내용

#### 튜닝 5: Encoder 변경 — ResNet34 → ResNet50

| | ResNet34 (Before) | ResNet50 (After) |
|---|---|---|
| Block 타입 | BasicBlock (conv 2개) | **Bottleneck (conv 3개)** |
| Expansion | 1 | **4** |
| 파라미터 수 | 21.8M | **25.6M** |
| 최종 채널 | 512 | **2048** |
| Layer 구성 | [3, 4, 6, 3] | [3, 4, 6, 3] |
| Stride | Layer1: 1, Layer2~4: 2 | Layer1: 1, Layer2~4: 2 |

**변경 이유**
- Bottleneck 구조의 expansion=4로 최종 채널이 512 → 2048로 4배 증가
- 더 풍부하고 세밀한 feature map을 Decoder에 전달
- 가느다란 스크래치, 미세한 이격 등 경계가 모호한 손상 패턴 캡처 능력 향상
- 모델 구조(U-Net)는 동일, encoder만 교체하므로 학습 파이프라인 변경 불필요

#### 튜닝 6: 입력 해상도 증가 — 256x256 → 512x512

| Before | After |
|---|---|
| 256 x 256 | **512 x 512** |

**변경 이유**
- 256x256에서 스크래치(1~3px 폭)는 거의 인식 불가 수준으로 축소됨
- 512x512로 올리면 동일 손상이 2~6px 폭으로 표현되어 검출 가능성 대폭 증가
- Recall 향상에 직접적 효과 기대

#### 튜닝 7: Batch Size 조정 — 64 → 32

| Before | After |
|---|---|
| 64 | **32** |

**변경 이유**
- 해상도 512x512는 256x256 대비 픽셀 수 4배 → GPU 메모리 사용량 4배 증가
- Batch Size를 64 → 32로 줄여 GPU 메모리 한계 내에서 안정적 학습 보장
- Batch Size 감소로 gradient noise가 약간 증가하나, 정규화 효과로 오히려 일반화 성능 향상 가능

### 2차 튜닝 전체 설정 비교

| 항목 | Baseline | 1차 튜닝 | 2차 튜닝 (최종) |
|---|---|---|---|
| **Encoder** | ResNet34 | ResNet34 | **ResNet50** |
| **입력 크기** | 256x256 | 256x256 | **512x512** |
| **Batch Size** | 64 | 64 | **32** |
| **Loss** | CrossEntropyLoss | DiceBCELoss | DiceBCELoss |
| **Augmentation** | 없음 | 5종 | 5종 |
| **LR Scheduler** | 없음 | CosineAnnealingLR | CosineAnnealingLR |
| **Encoder LR** | 1e-6 | 1e-5 | 1e-5 |
| **Decoder LR** | 3e-4 | 3e-4 | 3e-4 |
| **Weight Decay** | 0 | 1e-4 | 1e-4 |
| **Best 기준** | mIoU | F1 Score | F1 Score |
| **Early Stopping** | 없음 | patience=15 | patience=15 |

### 2차 튜닝 결과 (val set, Scratched 기준 — early stopping 전 중단)

| 모델 | Best F1 | Accuracy | mIoU | Best Epoch | 비고 |
|---|---|---|---|---|---|
| **Scratched** | 0.7209 | 96.90% | 0.7657 | 23 | 학습 중 중단 |
| **Separated** | 0.5562 | 99.03% | 0.6877 | 44 | 학습 중 중단 |
| **Crushed** | - | - | - | - | OOM으로 미완료 |
| **Breakage** | - | - | - | - | 미완료 |

### 2차 튜닝 분석

- Scratched F1: 0.6769 → **0.7209** (+0.044 개선) — ResNet50 + 512 해상도 효과 확인
- Recall: 0.5972 → **0.7198** — 놓치던 손상의 12%를 추가 검출
- **그러나 F1 0.9 목표에는 여전히 부족**

### 2차 튜닝 한계 원인

1. **입력 정규화 불일치**: ImageNet pretrained encoder를 사용하면서 ImageNet 정규화(mean/std)를 적용하지 않음 → Encoder가 기대하는 입력 분포와 실제 입력이 다름 → feature 추출이 최적이 아닌 상태
2. **단순 0~1 정규화**: `images / 255.`만 적용 → ImageNet에서 학습된 가중치가 기대하는 `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]` 분포와 불일치
3. **Augmentation 부족**: 블러 처리 등 실제 촬영 환경에서 발생하는 노이즈에 대한 강건성 미확보

---

## 4. 3차 튜닝 (ImageNet Normalize + Augmentation 보강)

2차 튜닝에서 발견된 입력 정규화 불일치 문제를 해결하고, Augmentation을 보강.

### 변경 내용

#### 튜닝 8: ImageNet Normalize 적용

| Before | After |
|---|---|
| `images / 255.` (단순 0~1 스케일링) | `A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])` |

**변경 이유**
- ResNet50은 ImageNet 데이터셋으로 사전학습됨
- ImageNet 학습 시 사용된 정규화: 각 채널을 `(pixel/255 - mean) / std`로 변환
- 기존에는 `/255.`만 적용하여 0~1 범위 → Encoder가 기대하는 분포와 불일치
- 정규화 적용 시 Encoder의 pretrained 가중치가 정상적으로 작동 → feature 추출력 즉시 향상

```
기존 입력 분포:    [0.0 ~ 1.0]      ← Encoder가 기대하는 범위가 아님
변경 입력 분포:    [-2.1 ~ 2.6]     ← ImageNet 학습 시와 동일한 분포
```

**기대 효과**: Encoder가 ImageNet에서 학습한 feature 추출 능력을 100% 활용 → F1 상승

#### 튜닝 9: GaussianBlur 증강 추가

| Before | After |
|---|---|
| 5종 증강 | **7종 증강** (+ Normalize, GaussianBlur) |

**추가된 증강:**

| 기법 | 설정 | 효과 |
|---|---|---|
| **GaussianBlur** | blur_limit=(3,7), p=0.2 | 실제 촬영 시 발생하는 초점 흐림/카메라 흔들림에 대한 강건성 |
| **A.Normalize** | ImageNet mean/std | 학습/검증 모두 동일한 정규화 보장 |

**변경 이유**: 실제 차량 촬영 환경에서는 카메라 초점 문제, 조명 반사 등으로 이미지가 흐릿할 수 있음. GaussianBlur를 학습 시 적용하면 이런 상황에서도 안정적으로 검출 가능.

**기대 효과**: 실환경 이미지에서의 일반화 성능 향상

#### 튜닝 10: 검증 데이터에도 동일 Normalize 적용

| Before | After |
|---|---|
| 검증 시 Resize만 적용 | 검증 시 **Resize + ImageNet Normalize** 적용 |

**변경 이유**: 학습과 검증의 입력 분포가 다르면 평가 결과가 왜곡됨. 학습/검증/추론 모두 동일한 정규화를 적용해야 정확한 성능 측정 가능.

### 3차 튜닝 전체 설정 비교

| 항목 | Baseline | 1차 튜닝 | 2차 튜닝 | 3차 튜닝 (현재) |
|---|---|---|---|---|
| **Encoder** | ResNet34 | ResNet34 | ResNet50 | ResNet50 |
| **입력 크기** | 256x256 | 256x256 | 512x512 | 512x512 |
| **Batch Size** | 64 | 64 | 16 | 16 |
| **Loss** | CrossEntropyLoss | DiceBCELoss | DiceBCELoss | DiceBCELoss |
| **Augmentation** | 없음 | 5종 | 5종 | **7종** |
| **Normalize** | /255. (단순) | /255. (단순) | /255. (단순) | **ImageNet mean/std** |
| **LR Scheduler** | 없음 | CosineAnnealingLR | CosineAnnealingLR | CosineAnnealingLR |
| **Encoder LR** | 1e-6 | 1e-5 | 1e-5 | 1e-5 |
| **Decoder LR** | 3e-4 | 3e-4 | 3e-4 | 3e-4 |
| **Weight Decay** | 0 | 1e-4 | 1e-4 | 1e-4 |
| **Best 기준** | mIoU | F1 Score | F1 Score | F1 Score |
| **Early Stopping** | 없음 | patience=15 | patience=15 | patience=15 |

### 3차 튜닝 결과 (val set)

| 모델 | Best F1 | Accuracy | mIoU | Best Epoch |
|---|---|---|---|---|
| **Scratched** | - | - | - | - |
| **Separated** | - | - | - | - |
| **Crushed** | - | - | - | - |
| **Breakage** | - | - | - | - |

> 3차 튜닝 학습 진행 중 — 완료 후 업데이트 예정

---

## 5. 전체 튜닝 과정 요약

```
Baseline (ResNet34 / 256x256 / CE Loss / 정규화: /255.)
  │  Scratched F1: 0.6540, Acc: 96.49%
  │
  │  문제: F1 낮음 (목표 0.9 대비 부족)
  │  원인: 클래스 불균형, Augmentation 없음, LR 고정
  │
  ▼
1차 튜닝 (+ DiceBCELoss / Augmentation 5종 / CosineAnnealingLR / Encoder LR 10x)
  │  Scratched F1: 0.6769 (+0.02)
  │
  │  문제: F1 소폭 개선에 그침
  │  원인: 256x256 해상도 한계, ResNet34 feature 한계
  │
  ▼
2차 튜닝 (+ ResNet50 Encoder / 512x512 / Batch 16)
  │  Scratched F1: 0.7209 (+0.04)
  │
  │  문제: F1 상승폭 둔화
  │  원인: ImageNet Normalize 미적용 → pretrained encoder 활용 부족
  │
  ▼
3차 튜닝 (+ ImageNet Normalize / GaussianBlur / Val Normalize)
  │  학습 진행 중...
  │
  ▼
최종 결과 (업데이트 예정)
```

### 튜닝 항목 전체 목록

| # | 튜닝 항목 | 변경 내용 | 적용 시점 | 이유 | 기대 효과 |
|---|---|---|---|---|---|
| 1 | **Loss 함수** | CE → DiceBCELoss | 1차 | 클래스 불균형에서 F1 직접 최적화 | Recall/Precision 균형 |
| 2 | **Augmentation** | 없음 → 5종 | 1차 | 학습 데이터 다양성 부족 | 과적합 방지, 일반화 |
| 3 | **LR Scheduler** | 고정 → CosineAnnealing | 1차 | 학습 후반 미세 조정 불가 | 수렴 안정화 |
| 4 | **Encoder LR** | 1e-6 → 1e-5 | 1차 | 도메인 특화 fine-tuning 부족 | feature 추출력 향상 |
| 5 | **Encoder** | ResNet34 → ResNet50 | 2차 | 최종 채널 512→2048, feature 부족 | 세밀한 feature 추출 |
| 6 | **해상도** | 256 → 512 | 2차 | 가는 손상 디테일 소실 | Recall 대폭 향상 |
| 7 | **Batch Size** | 64 → 16 | 2차 | GPU 메모리 대응 | 안정적 학습 |
| 8 | **Normalize** | /255. → ImageNet mean/std | 3차 | pretrained encoder 입력 분포 불일치 | encoder 활용 극대화 |
| 9 | **GaussianBlur** | 없음 → blur_limit=(3,7) | 3차 | 실환경 촬영 노이즈 미대응 | 실환경 강건성 |
| 10 | **Val Normalize** | Resize만 → Resize+Normalize | 3차 | 학습/검증 정규화 불일치 | 정확한 성능 측정 |

---

## 6. 성능 변화 추이 (Scratched 기준)

| 단계 | F1 Score | Accuracy | mIoU | Recall | 핵심 변경 |
|---|---|---|---|---|---|
| Baseline | 0.6540 | 96.49% | 0.7103 | 0.5972 | - |
| 1차 튜닝 | 0.6769 | 96.49% | 0.7376 | - | Loss/Aug/LR |
| 2차 튜닝 | 0.7209 | 96.90% | 0.7657 | 0.7198 | ResNet50/512 |
| 3차 튜닝 | - | - | - | - | ImageNet Norm |

> 3차 튜닝 결과는 학습 완료 후 업데이트 예정

---

## 7. 사용 커맨드

### Baseline 학습
```bash
cd "1.AI모델 소스코드/code"
python main.py --train train --task damage --label all
```

### 3차 튜닝 학습 (GPU당 순차 2개, 총 4개)
```bash
cd "1.AI모델 소스코드/code"

# GPU 0: Scratched → 끝나면 → Crushed 자동 시작
# GPU 1: Separated → 끝나면 → Breakage 자동 시작
(python train_tuned.py --label_idx 0 --label_name Scratched --gpu 0 --patience 15 && \
 python train_tuned.py --label_idx 2 --label_name Crushed --gpu 0 --patience 15) &
(python train_tuned.py --label_idx 1 --label_name Separated --gpu 1 --patience 15 && \
 python train_tuned.py --label_idx 3 --label_name Breakage --gpu 1 --patience 15) &
```

### 평가
```bash
cd "1.AI모델 소스코드/code"
python eval_metrics.py
```
