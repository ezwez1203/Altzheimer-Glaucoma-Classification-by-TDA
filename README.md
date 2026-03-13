# RetiQ — Retinal Vessel Analysis for Alzheimer & Glaucoma Classification

망막 혈관 분석 파이프라인: **Vascular Connectivity Prediction (VCP)**, 그래프 기반 혈관 추출, **Topological Data Analysis (TDA)**, 통계적 특징 선별, SVM 분류를 결합하여 녹내장 및 알츠하이머 스크리닝을 수행.

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
│   ├── main.py                  엔트리포인트
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
│   ├── data_loader.py           Track A+B CSV 병합, 라벨 할당
│   ├── stat_tests.py            Shapiro-Wilk → ANOVA / Kruskal-Wallis
│   ├── plots.py                 Boxplot + Correlation Heatmap
│   ├── selection.py             p < 0.05 특징 선별
│   ├── main.py                  엔트리포인트
│   ├── labels.csv               50 subjects 임상 라벨
│   ├── selected_features.csv    선별 특징 → 05_SVM 입력
│   ├── statistical_results.csv  전체 검정 결과
│   └── plots/                   학술 시각화
│
├── 05_SVM/                      Pipeline 기반 SVM 분류기
│   ├── svm_classifier.py        통합 분류 스크립트 (권장 엔트리포인트)
│   ├── config.py                경로, CV 파라미터, grid search 공간
│   ├── classifier.py            TopologicalSVMClassifier 클래스
│   ├── plots.py                 Confusion matrix, ROC 곡선
│   ├── main.py                  모듈형 엔트리포인트
│   └── results/
│       ├── confusion_matrix.png 혼동 행렬
│       ├── roc_curves.png       OVR ROC 곡선
│       ├── decision_boundary.png SVM 결정 경계 시각화
│       ├── fold_metrics.csv     fold별 성능 + mean ± std
│       └── best_svm_model.pkl   Pipeline + LabelEncoder
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

Track A + B에서 추출된 12개 특징에 대해:

- 정규성 검정 (Shapiro-Wilk) → ANOVA 또는 Kruskal-Wallis H-test
- 학술 논문 수준 시각화 (Boxplot + Swarmplot, Correlation Heatmap)
- p < 0.05 기준 유의미한 특징만 선별 → `selected_features.csv`

소규모 데이터셋(N=50)에서 12개 특징을 모두 사용하면 차원의 저주로 과적합이 발생하므로, 통계 검정으로 노이즈 특징을 사전 제거.

### Stage 5 — SVM Classifier (`05_SVM/`)

선별된 TDA 특징을 기반으로 Normal / Glaucoma / Alzheimer 3-class 분류:

- **Pipeline 기반 학습**: `StandardScaler → SVC`를 `sklearn.pipeline.Pipeline`으로 결합하여 scaling이 항상 CV fold 내부에서만 수행 → 데이터 누출 방지
- **Repeated Stratified K-Fold CV** (3회 반복 × 5-fold = 15회 평가): 소규모 데이터에서 단일 K-Fold보다 분산이 줄어든 안정적 성능 추정
- **GridSearchCV**: `C`, `kernel`, `gamma` 하이퍼파라미터 동시 튜닝 (scoring=`f1_macro`)
- **클래스 불균형 처리**: `class_weight='balanced'`로 소수 클래스(Glaucoma, N=7) 보상. N=50에서 SMOTE는 노이즈 증폭으로 과적합을 악화시키므로 비사용
- **평가 지표**: Accuracy, Sensitivity, Specificity, Precision, F1 (macro), AUROC (OVR macro)
- **시각화**: 혼동 행렬, ROC 곡선, 2D 결정 경계 (2개 특징이므로 가능)
- **출력**: fold별 성능 CSV (mean ± std 포함), 학습 완료 모델(.pkl)

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

# 2. Graph Extraction — 혈관 마스크 → 그래프
cd ../02_Graph_Extraction
conda activate cuda_tda
python main.py

# 3. Track A — 방사형 필트레이션 TDA (녹내장)
cd ../03_trackA
python main.py

# 4. Track B — Fractal Dimension + AVR + Flooding TDA (알츠하이머)
cd ../03_trackB
python main.py

# 5. EDA & Feature Selection — 통계 검정 + 시각화 + 특징 선별
cd ../04_EDA_and_Selection
python main.py

# 6. SVM Classification — Repeated Stratified K-Fold SVM
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
