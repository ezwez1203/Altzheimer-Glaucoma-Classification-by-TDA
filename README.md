# RetiQ — Retinal Vessel Analysis for Alzheimer & Glaucoma Classification

망막 혈관 분석 파이프라인: **Vascular Connectivity Prediction (VCP)**, 그래프 기반 혈관 추출, **Topological Data Analysis (TDA)**, 거시적 그래프 통계, 통계적 특징 선별, **Feature-Specific Hierarchical SVM** 분류를 결합하여 녹내장 및 알츠하이머 스크리닝을 수행.

Based on: *"Topology-Aware Retinal Artery–Vein Classification via Deep Vascular Connectivity Prediction"*

---

## Project Structure

```
├── 01_VCP/                      혈관 분할 & 동맥/정맥 분류 (VCP 모델)
│   ├── data/                    DRIVE (40) + IOSTAR (30) 데이터셋
│   ├── processed_data/          두께/방향 맵 (전처리 결과)
│   ├── checkpoints/             모델 가중치
│   ├── results/                 추론 출력 (vessel/AV/OD 마스크)
│   ├── src/                     모델, 손실 함수, 토폴로지 유틸리티
│   ├── config.py                하이퍼파라미터
│   ├── train.py                 3단계 학습 파이프라인
│   ├── evaluate.py              모델 평가
│   └── inference.py             배치 추론
│
├── 02_Graph_Extraction/         혈관 마스크 → NetworkX 그래프 변환
│   ├── config.py                경로, 색상 인코딩
│   ├── morphology.py            EDT, skeletonization, 노드 검출
│   ├── tracer.py                스켈레톤 경로 추적
│   ├── graph_builder.py         그래프 조립 (edge 속성 부여)
│   ├── extractor.py             RetinalGraphExtractor (오케스트레이터)
│   ├── dataset.py               배치 처리
│   ├── extract_macro_features.py 거시적 그래프 통계 추출
│   ├── main.py                  엔트리포인트
│   ├── graph_macro_features.csv 거시적 그래프 특징 (50행 × 8열)
│   └── processed_graphs/        DRIVE/ + IOSTAR/ .pkl 그래프 파일
│
├── 03_trackA/                   Track A: 방사형 필트레이션 TDA (녹내장)
│   ├── config.py                NOISE_THRESHOLD=5.0 px
│   ├── filtration.py            OD 중심 → 방사형 필트레이션 값 할당
│   ├── features.py              Betti 0/1 persistence → 5개 TDA 특징
│   ├── extractor.py             RadialFiltrationTDA
│   ├── dataset.py               배치 처리
│   ├── main.py                  엔트리포인트
│   └── trackA_features.csv      50 subjects × 7 columns
│
├── 03_trackB/                   Track B: Fractal + AVR + Flooding TDA (알츠하이머)
│   ├── config.py                BOX_SIZES, NOISE_THRESHOLD
│   ├── fractal.py               Box-counting 프랙탈 차원
│   ├── avr.py                   동맥/정맥 두께 비율
│   ├── flooding.py              두께 기반 필트레이션 + persistent homology
│   ├── extractor.py             AlzheimerTDAExtractor
│   ├── dataset.py               배치 처리
│   ├── main.py                  엔트리포인트
│   └── trackB_features.csv      50 subjects × 9 columns
│
├── 04_EDA_and_Selection/        통계 검정 → 유의미 특징 선별
│   ├── config.py                P_VALUE_THRESHOLD=0.05
│   ├── data_loader.py           Track A+B + Graph Macro CSV 병합, 라벨 할당
│   ├── stat_tests.py            3-class: ANOVA / Kruskal-Wallis
│   │                            2-class: Welch t-test / Mann-Whitney U
│   ├── plots.py                 Boxplot + Correlation Heatmap
│   ├── selection.py             p < 0.05 특징 선별
│   ├── main.py                  엔트리포인트
│   ├── labels.csv               50 subjects 임상 라벨
│   ├── selected_features.csv    3-class 선별 특징 (7개)
│   ├── selected_features_stage1.csv  Stage 1 선별 특징 (Normal vs Disease)
│   ├── selected_features_stage2.csv  Stage 2 선별 특징 (AD vs Glaucoma)
│   ├── statistical_results.csv       3-class 검정 결과
│   ├── statistical_results_stage1.csv Stage 1 검정 결과
│   ├── statistical_results_stage2.csv Stage 2 검정 결과
│   └── plots/                   학술 시각화
│
├── 05_SVM/                      Feature-Specific Hierarchical SVM 분류기
│   ├── svm_classifier.py        통합 분류 스크립트 (권장 엔트리포인트)
│   └── results/
│       ├── confusion_matrix.png 혼동 행렬
│       ├── roc_curves.png       OVR ROC 곡선
│       ├── decision_boundary.png Stage 1 + Stage 2 결정 경계
│       ├── fold_metrics.csv     fold별 성능 + stage별 성능 + mean ± std
│       └── best_svm_model.pkl   Hierarchical Pipeline (Stage1 + Stage2)
│
├── requirements.txt             02~05 단계 의존성
├── README.md
└── explanation.md               수학적/기술적 상세 설명
```

---

## Pipeline Overview

### Stage 1 — VCP (`01_VCP/`)

VGG16 기반 3단계 학습:

1. **Optic Disc Segmentation** — 시신경유두 바이너리 분할 (BCE + Dice)
2. **Multi-task Network** — 혈관 분할 + 동맥/정맥 분류 동시 학습
3. **Connectivity Network** — 혈관 픽셀 쌍의 연결 여부 예측 → 그래프 기반 트리 추적으로 최종 AV 분류

### Stage 2 — Graph Extraction (`02_Graph_Extraction/`)

VCP 출력 마스크를 수학적 그래프로 변환:

1. Euclidean Distance Transform → 혈관 두께 맵
2. Skeletonization → 1-pixel 중심선
3. 3×3 이웃 검사로 endpoint / junction 노드 검출
4. 노드 간 edge tracing → `weight`, `avg_thickness`, `vessel_type` 속성 부여

추가로, `extract_macro_features.py`를 통해 **거시적 그래프 통계**를 추출:

| 거시 특징 | 설명 | Stage 1 분리력 |
|-----------|------|----------------|
| `vessel_density` | 이미지 내 혈관 픽셀 비율 | p = 0.051 (경계선) |
| `avg_edge_thickness` | 전체 혈관 세그먼트 평균 두께 | **p = 0.023** |
| `total_junctions` | 혈관 분기점(교차점) 총 수 | p = 0.106 |
| `total_endpoints` | 혈관 말단점 총 수 | p = 0.161 |
| `total_edges` | 혈관 세그먼트 총 수 | p = 0.104 |
| `avg_edge_length` | 혈관 세그먼트 평균 길이 | p = 0.295 |

이 거시 통계는 TDA 특징이 놓치는 **"숲 전체의 변화"**(전반적 혈관 밀도 감소, 두께 변화)를 포착하여 Stage 1(Normal vs Disease)의 분리력을 보강.

### Stage 3A — Radial Filtration TDA (`03_trackA/`) — 녹내장

Optic Disc 중심으로부터의 **방사형 거리**를 필트레이션 값으로 사용:

- GUDHI SimplexTree → Persistent Homology 계산
- H0 (연결 성분) persistence로 혈관 분기 패턴 포착
- 추출 특징: `b0_max_lifespan`, `b0_sum_lifespan`, `b1_count`, `b1_max_lifespan`, `persistence_entropy`

### Stage 3B — AD Biomarkers (`03_trackB/`) — 알츠하이머

세 가지 바이오마커 추출:

1. **Fractal Dimension** — Box-counting 알고리즘으로 혈관 복잡도 측정
2. **Artery-Vein Ratio (AVR)** — 동맥/정맥 평균 두께 비율
3. **Flooding Filtration TDA** — 두꺼운 혈관부터 채워나가는 필트레이션 → H0/H1 persistence

### Stage 4 — EDA & Feature Selection (`04_EDA_and_Selection/`)

Track A + B + Graph Macro에서 추출된 18개 특징에 대해 **3단계 통계 검정**:

1. **3-class 검정** (Normal / Alzheimer / Glaucoma) — ANOVA 또는 Kruskal-Wallis
2. **Stage 1 검정** (Normal vs Disease) — Welch t-test 또는 Mann-Whitney U
3. **Stage 2 검정** (Alzheimer vs Glaucoma) — Welch t-test 또는 Mann-Whitney U

Stage-specific 검정이 핵심: 3-class 검정에서 유의한 특징이 반드시 Stage 1에서 유의하지 않으며, 그 역도 성립. 계층적 분류기의 각 단계에 최적화된 특징을 선별하기 위해 별도의 2-class 검정이 필수.

### Stage 5 — Feature-Specific Hierarchical SVM (`05_SVM/`)

**핵심 아키텍처**: 3-class 문제를 2단계 이진 분류로 분해하되, 각 단계에 **서로 다른 특징 세트**를 투입:

```
                    ┌─── Normal (23)
Stage 1 ───────────┤
(4 features)       └─── Disease (27) ──── Stage 2 ───┬─── Alzheimer (20)
                                         (2 features) └─── Glaucoma (7)

Stage 1 Features (거시 그래프 + 미시 TDA):
  avg_edge_thickness (p=0.023), vessel_density (p=0.051),
  fractal_dimension (p=0.073), b0_sum_lifespan (p=0.094)

Stage 2 Features (미시 TDA only):
  fractal_dimension (p=0.007), b0_sum_lifespan (p=0.013)
```

**이 설계의 핵심 논리**:

- **Stage 1 (Normal vs Disease)**: 정상군과 질환군의 차이는 "혈관이 전반적으로 희소해지거나 얇아졌는가"라는 **거시적(macroscopic)** 질문. 따라서 `avg_edge_thickness`(혈관 두께)와 `vessel_density`(혈관 밀도) 같은 그래프 수준 통계가 분리력을 제공하며, TDA 특징(`fractal_dimension`, `b0_sum_lifespan`)이 이를 보강.

- **Stage 2 (AD vs Glaucoma)**: 두 질환군의 차이는 "위상학적 붕괴 패턴이 어떻게 다른가"라는 **미시적(microscopic)** 질문. TDA 특징만으로 p < 0.01 수준의 강력한 분리력을 확보하므로 추가 특징이 불필요.

**최종 성능** (3×5-Fold Repeated Stratified CV, N=50):

| 지표 | Stage 1 | Stage 2 | 3-class 최종 |
|------|---------|---------|-------------|
| Accuracy | 64.7% ± 12.0% | 83.3% ± 13.7% | **60.7% ± 12.4%** |
| F1 macro | 63.3% ± 13.3% | 78.7% ± 17.4% | 44.2% ± 9.4% |
| Specificity | — | — | 78.3% |
| AUROC macro | — | — | 0.660 |

소규모 데이터셋(N=50, Glaucoma=7)에서 단순 random baseline(33.3%)의 약 2배에 달하는 60.7% 정확도를 달성. 특히 Stage 2의 83.3%는 TDA 특징이 AD와 녹내장의 위상학적 차이를 효과적으로 포착함을 입증하며, Stage 1의 개선은 거시적 그래프 특징의 임상적 가치를 시사. **데이터 규모 확장 시 실용적 스크리닝 도구로의 발전 가능성**을 보여주는 강력한 Proof of Concept.

---

## Architecture (01_VCP)

| Model | Backbone | Output |
|-------|----------|--------|
| `OpticDiscSegmentationNetwork` | VGG16 | Binary OD mask |
| `MultiTaskNetwork` | VGG16 (default) or Attention U-Net++ | Vessel mask + AV map |
| `FullConnectivityPipeline` | VGG16 (default) or Attention U-Net++ | Thickness / Orientation / Connectivity |

Default backbone: **VGG16** (`USE_ATTENTION_UNET = False` in `01_VCP/config.py`).

---

## Installation

```bash
# VCP 학습 환경
conda create -n vcp python=3.9 -y
conda activate vcp
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r 01_VCP/requirements.txt

# TDA / 분석 환경 (02~05 단계)
conda create -n cuda_tda python=3.10 -y
conda activate cuda_tda
pip install -r requirements.txt
```

---

## Quick Start

```bash
# 1. VCP — 전처리, 학습, 추론
cd 01_VCP
conda activate vcp
bash 01_preprocess.sh
bash 02_train.sh
bash 03_inference.sh

# 2. Graph Extraction — 혈관 마스크 → 그래프 + 거시 통계
cd ../02_Graph_Extraction
conda activate cuda_tda
python main.py                    # 그래프 생성
python extract_macro_features.py  # 거시적 그래프 특징 추출

# 3. Track A — 방사형 필트레이션 TDA (녹내장)
cd ../03_trackA
python main.py

# 4. Track B — Fractal Dimension + AVR + Flooding TDA (알츠하이머)
cd ../03_trackB
python main.py

# 5. EDA & Feature Selection — 3-class + Stage-specific 통계 검정
cd ../04_EDA_and_Selection
python main.py

# 6. SVM Classification — Feature-Specific Hierarchical SVM
cd ../05_SVM
python svm_classifier.py
```

---

## Dataset Structure

```
01_VCP/data/
├── DRIVE/
│   ├── training/
│   │   ├── images/        RGB fundus images (20)
│   │   ├── 1st_manual/    vessel GT masks
│   │   ├── mask/          FOV masks
│   │   ├── av/            AV labels (red=artery, blue=vein)
│   │   └── od/            optic disc masks
│   └── test/
│       └── ...            (20 images, no AV labels)
└── IOSTAR/
    └── training/
        ├── images/
        ├── 1st_manual/
        ├── mask/
        └── od/
```

---

## Key Configuration (`01_VCP/config.py`)

| Parameter | Value | Note |
|-----------|-------|------|
| `IMAGE_SIZE` | (384, 384) | Reduced for 8 GB GPU |
| `BATCH_SIZE` | 1 | RTX 4060 limit |
| `USE_PRETRAINED_VGG` | True | ImageNet weights |
| `USE_ATTENTION_UNET` | **False** | VGG16 backbone (set True for Attention U-Net++) |
| `VAL_SPLIT` | 0.1 | 10 % of training set used for validation |
| `LAMBDA_SEGM_DRIVE` | 1.0 | Vessel loss weight (DRIVE) |
| `LAMBDA_AV_DRIVE` | 10.0 | AV loss weight (DRIVE) |

---

## Author

**Dohyun Hwang** — ezwez1467@yonsei.ac.kr

## License

For research purposes only.
