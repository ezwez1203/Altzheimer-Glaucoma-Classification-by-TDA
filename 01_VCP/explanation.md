# VCP 프로젝트 코드 설명

> 브랜치: `feature/enhanced-vcp`
> 최종 수정: 2026-03-09

---

## 1. 전체 구조

이 프로젝트는 논문 *"Topology-Aware Retinal Artery–Vein Classification via Deep Vascular Connectivity Prediction"* 을 구현한 망막 혈관 동맥/정맥 분류 파이프라인이다.

```
data/
├── DRIVE/          (20 train + 20 test)
└── IOSTAR/         (30 train)

processed_data/
├── thickness_maps/
└── orientation_maps/

src/
├── models.py       모델 구조 전체
├── dataset.py      데이터 로딩
├── losses.py       손실 함수
├── preprocessing.py 두께/방향 맵 생성
├── topology.py     트리 추적 / AV 분류
└── graph_utils.py  혈관 그래프 구성

train.py            학습 진입점
inference.py        추론
evaluate.py         평가
config.py           하이퍼파라미터
```

---

## 2. 학습 파이프라인

### 2-1. 전체 흐름 (3단계)

```
[1단계] Optic Disc Segmentation
  → VGG16 인코더로 시신경 원판(OD) 바이너리 분할
  → 손실: BCE + Dice

[2단계] Multi-task Network
  → VGG16 + MultiScaleFeatureFusion
  → 출력 헤드 2개:
      vessel_decoder: 혈관/배경 이진 분할
      av_decoder:    Artery(1) / Vein(2) / Background(0) 분류
  → 손실: λ_segm × L_vessel + λ_AV × L_AV
      DRIVE: λ_segm=1.0, λ_AV=10.0
      IOSTAR: λ_segm=0.01, λ_AV=1.0

[3단계] Connectivity Network (2-step)
  Step1 (λ_conn=0): 두께(5 class) + 방향(7 class) 보조 태스크만 학습
  Step2 (전체):     혈관 위 임의 64쌍 좌표 샘플링 → 연결 여부 이진 분류
  → 손실: λ_thick×L_thick + λ_ori×L_ori + λ_conn×L_conn
          100 × 100 × 1 (default)
```

### 2-2. 실행 방법

```bash
# 전처리 (두께/방향 맵 생성) - 병렬 처리
bash 01_preprocess.sh

# 학습
python train.py --dataset DRIVE --task all
python train.py --dataset DRIVE --task multitask --epochs 100

# 추론
python inference.py --dataset DRIVE

# 평가
python evaluate.py --dataset DRIVE
```

---

## 3. 변경 사항 이력

### 3-1. VGG16 백본으로 전환 (`config.py`, `src/models.py`, `train.py`)

**배경**: 기존 코드의 기본 백본이 Attention U-Net++(`use_attention_unet=True`)였음.
논문 원본은 VGG16 기반이므로 기본값을 VGG16으로 되돌림.

**변경 파일**

| 파일 | 변경 내용 |
|------|-----------|
| `config.py` | `USE_ATTENTION_UNET = False` 추가 |
| `src/models.py` | `MultiTaskNetwork`, `ThicknessOrientationEncoder`, `FullConnectivityPipeline` 세 클래스의 `use_attention_unet` 기본값 `True → False` |
| `train.py` | `MultiTaskNetwork(... use_attention_unet=config.USE_ATTENTION_UNET)`, `FullConnectivityPipeline(... use_attention_unet=config.USE_ATTENTION_UNET)` |

Attention U-Net++로 되돌리려면 `config.py`에서 `USE_ATTENTION_UNET = True`로 변경.

---

### 3-2. F1 Score 계산 + TXT 로그 (`train.py`)

**배경**: 기존에는 loss만 TensorBoard에 기록하고, F1 계산과 txt 파일 로그가 없었음.

**추가된 헬퍼 함수**

```python
def compute_f1(preds, targets, num_classes):
    # background(0) 제외, 클래스별 TP/FP/FN으로 직접 계산 (sklearn 불필요)
    # 반환: [f1_class1, f1_class2, ...]

def open_log_file(config, prefix, dataset_name):
    # results/logs/{prefix}_{dataset}_{timestamp}.txt 생성
    # 파일 핸들과 경로 반환
```

**각 학습 함수별 변경**

| 함수 | 추가 내용 |
|------|-----------|
| `train_optic_disc_segmentation` | validation F1 (OD binary), txt 로그 (`epoch / train_loss / val_loss / f1_od`) |
| `train_multitask_network` | validation F1 (artery / vein / macro), TensorBoard `F1/artery`, `F1/vein`, `F1/macro`, txt 로그 |
| `train_connectivity_network` | txt 로그 (`step / epoch / train_loss`) |

**로그 파일 위치**

```
results/logs/
├── od_segm_DRIVE_20260309_120000.txt
├── multitask_DRIVE_20260309_120000.txt
├── connectivity_DRIVE_20260309_120000.txt
└── multitask_DRIVE_20260309_120000/   ← TensorBoard 이벤트
```

TensorBoard 확인: `tensorboard --logdir results/logs`

---

### 3-3. 전처리 속도 개선 (`src/preprocessing.py`)

**병목 1: `_order_centerline_coords` — O(N²) Python 루프**

기존 코드는 centerline 픽셀 순서를 정할 때 `remaining` set을 Python 루프로 순회 → N픽셀마다 N번 반복.

```python
# 변경 전: Python 루프 O(N²)
for idx in remaining:
    dist = np.sum((coords[idx] - current) ** 2)
    ...

# 변경 후: numpy 벡터화 O(N²)이지만 numpy C 코드 실행 → ~100배 빠름
diff = coords - current        # [N, 2] 한 번에 계산
dists = diff[:,0]**2 + diff[:,1]**2
nearest_idx = int(dists.argmin())
```

**병목 2: `batch_generate_maps` — 단일 프로세스 직렬 처리**

```python
# 변경 후: multiprocessing.Pool로 CPU 코어 수만큼 병렬 처리
with Pool(processes=num_workers) as pool:
    list(tqdm(pool.imap(worker_fn, mask_files), ...))
```

DRIVE 20장 + IOSTAR 30장 = 50장을 CPU 코어 수만큼 동시에 처리.

---

### 3-4. 코드 전반 최적화

#### `src/losses.py`
- `WeightedCrossEntropyLoss.class_weights` → `register_buffer`로 등록
  → forward마다 `.to(device)` 호출 제거
- `BinaryCrossEntropyLoss.pos_weight` → `register_buffer`로 등록
  → forward마다 `torch.tensor([...]).to(device)` 제거

#### `src/dataset.py`
- `__init__`에서 `_resolve_aux_paths()`로 thickness/orientation 경로 미리 계산
  → 샘플당 2~4번 `os.path.exists()` 시스템 콜 제거

#### `src/models.py` — `ConnectivityNetwork`
- `forward`: batch 루프 → tensor 고급 인덱싱
  ```python
  # 변경 전: for b in range(B): feat_i = combined_features[b, :, yi, xi]
  # 변경 후:
  b_idx = torch.arange(B, device=...)
  features_i = combined_features[b_idx, :, yi, xi]  # Python 루프 없음
  ```
- `forward_batch`: B×N 루프 → flatten 후 단일 행렬 연산

#### `train.py` — connectivity 쌍 샘플링
```python
# 변경 전: for p in range(64): ... (64번 Python 반복)
# 변경 후:
idx_i = torch.randint(0, num_points, (num_pairs,))  # 벡터로 한 번에
connectivity_gt[b] = (dy*dy + dx*dx < 400).float()  # sqrt 없이 제곱 비교
```

#### `src/topology.py`
- `compute_segment_thickness`: coord별 append 루프 → numpy 인덱싱 `thickness_map[:, ys, xs]`
- `tree_tracing`: 매 iteration마다 전체 connectivity dict 스캔 → adjacency list 사전 빌드로 O(N×|E|) → O(|E|)
- `classify_trees`: 픽셀별 append 루프 → `np.vstack` + numpy 인덱싱
- `visualize_topology`: coord loop → numpy 인덱싱 + color_array 미리 변환

#### `src/graph_utils.py`
- `extract_vessel_segments` junction 탐지: 3중 Python 루프 → `pixel_to_vertex` 룩업 테이블 + numpy 8방향 스캔
- `VesselGraph.get_neighbors`: 매 호출마다 edge 전체 스캔 O(|E|) → adjacency dict 캐싱 O(1)
- `create_segment_label_map`: coord loop → numpy 고급 인덱싱

---

## 4. 모델 구조 상세

### MultiTaskNetwork

```
Input [B, 3, H, W]
    ↓
VGG16Encoder (or AttentionUNetPlusPlusEncoder)
    ↓
MultiScaleFeatureFusion (5개 스케일 → H/2×W/2로 bilinear resize 후 concat)
    ↓
┌─────────────────┐
│ vessel_decoder  │ → [B, 1, H, W]  (BCE + Dice)
│ av_decoder      │ → [B, 3, H, W]  (CrossEntropy)
└─────────────────┘
```

### FullConnectivityPipeline

```
Input [B, 3, H, W]
    ↓
ThicknessOrientationEncoder (VGG16 기반)
    → thick_logits [B, 5, H, W]
    → ori_logits   [B, 7, H, W]
    → penultimate_features [B, C, H, W]
    ↓
ConnectivityNetwork (coords_i, coords_j 입력 시)
    → 두 좌표의 feature 추출 → FC → Hadamard product → FC → [B, N, 2]
```

---

## 5. 설정값 (config.py)

| 항목 | 값 | 설명 |
|------|----|------|
| `IMAGE_SIZE` | (384, 384) | RTX 4060 8GB 기준 축소 |
| `BATCH_SIZE` | 1 | GPU 메모리 제한 |
| `NUM_EPOCHS_STEP1` | 100 | OD / Multitask 학습 epoch |
| `NUM_EPOCHS_STEP2` | 100 | Connectivity Step2 epoch |
| `USE_PRETRAINED_VGG` | True | ImageNet 사전학습 가중치 |
| `USE_ATTENTION_UNET` | **False** | VGG16 사용 (True = Attention U-Net++) |
| `VAL_SPLIT` | 0.1 | 학습셋의 10%를 validation으로 사용 |

---

## 6. 체크포인트 및 결과 경로

```
checkpoints/
├── od_segm_{dataset}_best.pth
├── multitask_{dataset}_best.pth
├── connectivity_{dataset}_step1.pth
└── connectivity_{dataset}_best.pth

results/
├── logs/
│   ├── *.txt          ← 에포크별 loss / F1 로그
│   └── */             ← TensorBoard 이벤트
├── DRIVE/
│   └── {id}_test_{type}.png   (vessel, av_pixelwise, av_treewise, od, orientation, thickness, topology)
└── IOSTAR/
    └── {name}_{type}.png
```
