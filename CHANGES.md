# CHANGES.md — 전체 작업 기록

---

# PART 1 — Python 3.12 호환 업그레이드 + 실행 오류 해결

## Step 1. requirements.txt — 패키지 버전 업그레이드

| 패키지 | 기존 버전 | 변경 버전 | 변경 이유 |
|---|---|---|---|
| `numpy` | (명시 없음) | `1.26.4` | Python 3.12는 numpy 1.24 미만과 호환 불가. C API 변경으로 명시 추가 필요 |
| `pandas` | `1.3.4` | `2.2.2` | pandas 1.x는 Python 3.12 미지원. 2.0+ 부터 3.12 공식 지원 |
| `opencv-python` | `4.5.4.60` | `4.10.0.84` | 4.5.x는 Python 3.12 빌드 휠 미제공. 4.8+ 부터 3.12 지원 |
| `torch` | `1.10.1` | `2.4.0` | torch 1.x는 Python 3.12 미지원. 2.0+ 부터 3.12 공식 지원 |
| `segmentation-models-pytorch` | `0.2.1` | `0.3.3` | torch 2.x API 변경에 맞춰 업데이트된 버전 필요 |
| `albumentations` | `1.1.0` | `1.4.3` | `Cutout` 클래스 제거 및 파라미터 변경. Python 3.12 지원 |
| `pytz` | `2021.3` | `2024.1` | 최신 시간대 데이터 반영 |
| `tqdm` | (누락) | `4.66.4` | `Evaluation.py`에서 `import tqdm` 사용 중이나 requirements에 누락되어 있었음 |
| `pycocotools` | (conda 전용) | `2.0.8` | Dockerfile에서만 conda로 설치됨. pip 환경에서도 설치 가능하도록 추가 |

---

## Step 2. main.py — `A.Cutout` → `A.CoarseDropout`

**위치:** `main.py` (part 학습 transform 설정)

```python
# 기존
A.Cutout(p=0.3, max_h_size=32, max_w_size=32)

# 변경
A.CoarseDropout(p=0.3, max_height=32, max_width=32, max_holes=8)
```

**이유:** albumentations 1.4.x에서 `Cutout` 클래스와 파라미터(`max_h_size`, `max_w_size`)가 제거됨. 대체 클래스 `CoarseDropout`의 파라미터명은 `max_height`, `max_width`로 변경됨.

---

## Step 3. main.py / Evaluation.py — `torch.load()` `weights_only=False` 추가

**위치:** `main.py` load_model(), `Evaluation.py` load_model()

```python
# 기존
torch.load(weight_path, map_location=torch.device(device))

# 변경
torch.load(weight_path, map_location=torch.device(device), weights_only=False)
```

**이유:** PyTorch 2.x에서 `weights_only` 파라미터 미지정 시 `FutureWarning` 발생. 기존 `.pt` 파일 로드를 유지하려면 `False`로 명시해야 함.

---

## Step 4. Train.py / Evaluation.py — `torch.tensor(tuple)` → `torch.from_numpy(np.stack(...))`

**위치:** `Train.py` 학습/검증 루프, `Evaluation.py` 평가 루프

```python
# 기존
images = torch.tensor(images).float().to(self.device)
masks  = torch.tensor(masks).long().to(self.device)

# 변경
images = torch.from_numpy(np.stack(images)).float().to(self.device)
masks  = torch.from_numpy(np.stack(masks)).long().to(self.device)
```

**이유:** `collate_fn`이 `tuple(zip(*batch))`를 반환하므로 `images`는 numpy 배열의 tuple. PyTorch 2.x에서 tuple을 `torch.tensor()`에 직접 넘기는 방식은 경고 대상. `np.stack()`으로 먼저 단일 배열로 변환 후 `from_numpy()` 사용.

---

## PART 1 실행 오류 해결 플로우

### [오류 1] `RuntimeError: Found no NVIDIA driver on your system`

**발생 위치:** `main.py` line 46

```
RuntimeError: Found no NVIDIA driver on your system.
```

**원인:**
`torch.cuda.current_device()`를 GPU 유무 확인 없이 무조건 호출. 또한 코드 전체에서 `device = "cuda"`가 하드코딩되어 있어 GPU 없는 환경에서 전면 실패.

**수정 내용 (`main.py`):**
```python
# 기존 — GPU 무조건 호출
device = "cuda"

# 변경 — CPU/GPU 자동 감지
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'using device: {device}')
if device == 'cuda':
    print(f'gpu device num: {torch.cuda.current_device()}')
```

---

### [오류 2] `FileNotFoundError: ../data/datainfo/damage_Scratched_train.json`

**원인:** 학습에 필요한 COCO 포맷 JSON 파일 없음. 전처리 단계 미수행.

**수정 내용:** `setup_data.py` 신규 작성
1. `data/` 디렉토리 구조 생성
2. `New_Sample/` 데이터 심볼릭 링크
3. `damage_labeling.csv` 재생성 (train 80% / val 10% / test 10%)
4. COCO JSON 변환 자동 실행

---

### [오류 3] `Exception: input type is not supported` (pycocotools)

**원인:** segmentation 좌표 포맷이 COCO 표준과 다름

| 구분 | 포맷 |
|---|---|
| COCO 표준 | `[[x1, y1, x2, y2, ...]]` |
| 원본 데이터 | `[[[[x,y],[x,y],...]]]` |

**수정 내용 (`src/Utils.py` `rebuilding()`):**
```python
# 변경 — 재귀 flatten으로 어떤 중첩 깊이도 처리
def _flatten_numbers(obj):
    if isinstance(obj, (int, float)):
        return [obj]
    result = []
    for item in obj:
        result.extend(_flatten_numbers(item))
    return result

a['segmentation'] = [_flatten_numbers(a['segmentation'])]
```

---

### [오류 4] `TypeError: unsupported operand type(s) for %: 'NoneType' and 'int'`

**원인:** 인자 없이 실행 시 `n_cls = None` → `smp.Unet(classes=None)` → TypeError

**수정 내용 (`main.py`):**
```python
elif arg.cls is not None:
    n_cls = arg.cls
else:
    print("[오류] --task 또는 --cls 인자가 필요합니다.")
    exit(1)
```

---

### [결과] 정상 실행 확인

4개 damage 타입(Scratched / Separated / Crushed / Breakage) 모두 2 epoch씩 정상 학습 완료.
- 가중치: `data/weight/Unet_damage_label*_...pt`
- 로그: `data/result_log/[damage_label*]train_log.json`

---

## 수정된 파일 목록 (PART 1)

| 파일 | 수정 내용 |
|---|---|
| `code/requirements.txt` | Python 3.12 호환 버전 업그레이드, tqdm/pycocotools 추가 |
| `code/main.py` | CUDA 자동 감지, CoarseDropout 변경, weights_only=False, n_cls None 방어 처리 |
| `code/src/Train.py` | torch.tensor → from_numpy(np.stack) |
| `code/src/Evaluation.py` | weights_only=False, map_location 하드코딩 제거, from_numpy(np.stack) |
| `code/src/Utils.py` | segmentation 좌표 재귀 flatten |
| `setup_data.py` | 신규 작성 — 디렉토리 구조 생성, CSV 재생성, COCO JSON 변환 자동화 |

---

## Step 5. main.py — epochs 정리 및 100으로 변경

**위치:** `1.AI모델 소스코드/code/main.py`

```python
# 기존
epochs = [1,8,5,9]       # 사용되지 않는 리스트
# epochs = epochs[i],    # 주석처리된 코드
epochs = 2,              # 하드코딩

# 변경
epochs = 100,
```

**이유:** 테스트용 2 epoch에서 실제 학습을 위해 100 epoch으로 변경. 사용되지 않는 `epochs` 리스트와 주석처리된 코드도 함께 제거. 데이터 수집 완료 후 본격적인 모델 학습.

---

## Step 6. setup_data.py — New_Sample + 160.차량파손 이미지 데이터 통합

**위치:** `1.AI모델 소스코드/setup_data.py`

**변경 내용:**
- 기존: `New_Sample` 하나만 심볼릭 링크
- 변경: `New_Sample` + `160. 차량파손 이미지 데이터` 두 소스를 합쳐서 `data/Dataset/1.원천데이터/damage/`에 개별 파일 심볼릭 링크로 통합
- 중복 파일명은 자동 스킵
- CSV 및 COCO JSON도 통합된 데이터 기준으로 재생성

**이유:** 실제 학습에 AIHub 전체 데이터(402,103장 + New_Sample)를 모두 사용하기 위함.

---

---

# PART 2 — AI 모델 평가 + ai-army 연동 (2026-04-07)

## 1. AI 모델 평가 실행

**실행 명령어**:
```bash
cd "/home/hi/Downloads/workspace/1.AI모델 소스코드/code"
/home/hi/Downloads/workspace/.venv/bin/python main.py --eval y --task damage --dataset val
```

**오류 — 가중치 파일명 불일치**:
```
FileNotFoundError: ../data/weight/[DAMAGE][Scratch_0]Unet.pt
```
- 원인: `main.py`에 하드코딩된 파일명과 실제 저장된 파일명이 다름
- 해결: `main.py` line 171 `weight_paths` 수정 (각 label별 IoU 최고 파일로 교체)

```python
# 변경 후
weight_paths = ["../data/weight/"+n for n in [
    "Unet_damage_label0_start:2026-04-07 15:03:09 KST+0900_2_epoch_IoU_2e+01.pt",
    "Unet_damage_label1_start:2026-04-07 15:01:00 KST+0900_2_epoch_IoU_0.9.pt",
    "Unet_damage_label2_start:2026-04-07 15:02:24 KST+0900_2_epoch_IoU_0e+00.pt",
    "Unet_damage_label3_start:2026-04-07 15:03:41 KST+0900_2_epoch_IoU_0.7.pt"
]]
```

**평가 결과**:
| 손상 유형 | Loss | mIoU | Target IoU |
|-----------|------|------|------------|
| Scratched | 0.3094 | 0.5602 | 0.1754 |
| Separated | 0.1523 | 0.5004 | 0.0088 |
| Crushed   | 0.1509 | 0.4900 | 0.0000 |
| Breakage  | 0.1284 | 0.4960 | 0.0069 |

→ 학습 에폭 2회뿐이라 Target IoU 낮음. 추가 학습 필요.

---

## 2. FastAPI 서버 구축

**생성 파일**:
```
workspace/server/
├── main.py       — FastAPI 서버 (POST /inspect, GET /health)
├── inference.py  — 4개 UNet 모델 로드 + 추론
├── grading.py    — 마스크 → bbox 변환 + 등급(A/B/C/D) 계산
└── requirements.txt
```

**서버 실행 방법**:
```bash
cd /home/hi/Downloads/workspace/server
/home/hi/Downloads/workspace/.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
```

**오류 — 포트 충돌**:
```
ERROR: [Errno 98] address already in use
```
- 원인: 이전 서버 프로세스가 8000 포트 점유 중
- 해결: `fuser -k 8000/tcp` 후 재실행

**API 엔드포인트**:
- `GET /health` — 서버 및 모델 로드 상태 확인
- `POST /inspect` — 이미지 업로드 → 손상 판정 결과 반환

**응답 형식**:
```json
{
  "inspection_id": "uuid",
  "overall_grade": "A|B|C|D",
  "severity_score": 0.0~1.0,
  "inference_ms": 663,
  "detections": [
    { "class_id": 0, "class_name": "Scratched", "confidence": 0.82, "bbox": [x1, y1, x2, y2] }
  ],
  "created_at": "ISO8601"
}
```

---

## 3. ai-army 웹앱 연결

**ai-army 실행 방법**:
```bash
cd /home/hi/Downloads/workspace/ai-army/public
python3 -m http.server 3001
# 브라우저: http://localhost:3001
```

- 3000번 포트는 node 프로세스 점유 중 → 3001 사용

**연결 설정 순서**:
1. 서버 먼저 실행 (`uvicorn main:app --port 8000`)
2. 브라우저에서 `http://localhost:3001` 접속
3. 하단 ⚙️ 설정 탭 클릭
4. API URL 입력: `http://localhost:8000`
5. Mock 모드 해제 → 실제 AI 서버 연결

---

## 4. Intel RealSense 카메라 오류

**오류**: 카메라 접근불가

**원인**: `camera.js`의 `facingMode: "environment"`가 모바일 후면 카메라 전용 설정이라 USB 카메라(RealSense)에서 충돌

**수정 내용 (`ai-army/public/camera.js`)**:
```javascript
// 변경 후: 모바일/데스크탑 분기 + 실패 시 기본 설정으로 재시도
const isMobile = /Android|iPhone|iPad/i.test(navigator.userAgent);
const constraints = isMobile
  ? { video: { facingMode: { ideal: "environment" }, width: { ideal: 1920 }, height: { ideal: 1080 } }, audio: false }
  : { video: { width: { ideal: 1920 }, height: { ideal: 1080 } }, audio: false };

// 실패 시 fallback
this.stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
```

---

## 전체 시스템 구조

```
[사용자 브라우저 http://localhost:3001]
        ↓ POST /inspect (이미지)
[FastAPI 서버 http://localhost:8000]
        ↓
[UNet 모델 x4: Scratched / Separated / Crushed / Breakage]
        ↓ 세그멘테이션 마스크 → bbox + 등급
[JSON 응답]
        ↓
[ai-army 웹앱 결과 화면: 등급(A/B/C/D) + 손상 위치 표시]
```
