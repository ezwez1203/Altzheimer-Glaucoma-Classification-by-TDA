# RetiQ 프로젝트 코드 설명

> 최종 수정: 2026-03-13

---

## 1. 전체 구조

이 프로젝트는 논문 *"Topology-Aware Retinal Artery–Vein Classification via Deep Vascular Connectivity Prediction"* 을 기반으로 한 망막 혈관 분석 파이프라인이다. VCP 모델로 혈관을 분할한 뒤, 그래프 추출 → 거시적 그래프 통계 + TDA 특징 추출 → 통계 검정 → Feature-Specific Hierarchical SVM 분류까지의 전체 흐름을 구현.

```
01_VCP/
├── data/                   DRIVE (20+20) + IOSTAR (30)
├── processed_data/         두께/방향 맵
├── src/                    모델, 데이터셋, 손실 함수, 전처리, 토폴로지
├── train.py                학습 진입점
├── inference.py            추론
├── evaluate.py             평가
└── config.py               하이퍼파라미터

02_Graph_Extraction/
├── config.py               경로 상수, 색상 인코딩, 기본 파라미터
├── morphology.py           Distance transform, skeletonization, node detection
├── tracer.py               Skeleton 경로를 따라 edge tracing
├── graph_builder.py        NetworkX graph 조립 + AV label 처리
├── extractor.py            RetinalGraphExtractor (파이프라인 오케스트레이터)
├── dataset.py              파일 매칭, 배치 처리, I/O
├── extract_macro_features.py 거시적 그래프 통계 추출 (vessel density 등)
├── main.py                 엔트리포인트
├── graph_macro_features.csv 거시적 그래프 특징 (50행 × 8열)
└── processed_graphs/       {DRIVE,IOSTAR}/{id}_graph.pkl

03_trackA/
├── config.py               경로, noise threshold
├── filtration.py           OD center, 방사형 filtration, SimplexTree
├── features.py             Persistence diagram → TDA 특징
├── extractor.py            RadialFiltrationTDA
├── dataset.py              배치 처리, CSV 출력
├── main.py                 엔트리포인트
└── trackA_features.csv     50 subjects × 7 columns

03_trackB/
├── config.py               경로, box sizes, noise threshold
├── fractal.py              Box-counting fractal dimension
├── avr.py                  Artery-Vein Ratio
├── flooding.py             Flooding filtration + persistent homology
├── extractor.py            AlzheimerTDAExtractor
├── dataset.py              배치 처리, CSV 출력
├── main.py                 엔트리포인트
└── trackB_features.csv     50 subjects × 9 columns

04_EDA_and_Selection/
├── config.py               경로, p-value threshold, stage별 출력 경로
├── data_loader.py          Track A + B + Graph Macro CSV 병합, 라벨 처리
├── stat_tests.py           3-class: ANOVA / Kruskal-Wallis
│                           2-class: Welch t-test / Mann-Whitney U
├── plots.py                Boxplot, Correlation Heatmap
├── selection.py            유의미한 특징 선별
├── main.py                 엔트리포인트
├── statistical_results.csv       3-class 검정 결과
├── statistical_results_stage1.csv Stage 1 검정 결과 (Normal vs Disease)
├── statistical_results_stage2.csv Stage 2 검정 결과 (AD vs Glaucoma)
├── selected_features.csv         3-class 선별 특징 (7개)
├── selected_features_stage1.csv  Stage 1 선별 특징
├── selected_features_stage2.csv  Stage 2 선별 특징
└── plots/                  boxplots.png, correlation_heatmap.png

05_SVM/
├── svm_classifier.py       Feature-Specific Hierarchical SVM (통합 엔트리포인트)
└── results/
    ├── confusion_matrix.png    혼동 행렬
    ├── roc_curves.png          OVR ROC 곡선
    ├── decision_boundary.png   Stage 1 + Stage 2 결정 경계
    ├── fold_metrics.csv        fold별 + stage별 성능 표 (mean ± std)
    └── best_svm_model.pkl      Hierarchical Pipeline (Stage1 + Stage2 모델)
```

---

## 2. VCP 학습 파이프라인 (01_VCP)

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
bash 01_preprocess.sh       # 두께/방향 맵 생성 (병렬 처리)
bash 02_train.sh            # 전체 학습
bash 03_inference.sh        # 추론
```

---

## 3. 변경 사항 이력

### 3-1. VGG16 백본으로 전환

**배경**: 기존 기본 백본이 Attention U-Net++였으나, 논문 원본은 VGG16 기반이므로 기본값을 VGG16으로 되돌림.

| 파일 | 변경 내용 |
|------|-----------|
| `config.py` | `USE_ATTENTION_UNET = False` 추가 |
| `src/models.py` | 세 클래스의 `use_attention_unet` 기본값 `True → False` |
| `train.py` | `config.USE_ATTENTION_UNET` 참조 |

### 3-2. F1 Score 계산 + TXT 로그

기존에는 loss만 TensorBoard에 기록. 추가된 내용:

| 함수 | 추가 내용 |
|------|-----------|
| `train_optic_disc_segmentation` | validation F1, txt 로그 |
| `train_multitask_network` | artery/vein/macro F1, TensorBoard, txt 로그 |
| `train_connectivity_network` | txt 로그 |

### 3-3. 전처리 속도 개선

- `_order_centerline_coords`: Python 루프 → numpy 벡터화 (~100배 속도 향상)
- `batch_generate_maps`: 직렬 → `multiprocessing.Pool` 병렬 처리

### 3-4. 코드 전반 최적화

- `src/losses.py`: `register_buffer` 사용으로 forward시 `.to(device)` 제거
- `src/dataset.py`: 경로 미리 계산으로 `os.path.exists()` 시스템 콜 제거
- `src/models.py`: batch 루프 → tensor 고급 인덱싱
- `src/topology.py`: coord 루프 → numpy 인덱싱, adjacency list 사전 빌드
- `src/graph_utils.py`: junction 탐지 룩업 테이블, adjacency dict 캐싱

---

### 3-5. Graph Extraction 모듈 (`02_Graph_Extraction/`)

**배경**: VCP 출력 마스크를 TDA 분석을 위한 수학적 그래프로 변환.

#### Euclidean Distance Transform (EDT)으로 두께 추정

바이너리 vessel mask에서 각 혈관 픽셀 $p$에 대해 가장 가까운 배경 픽셀까지의 Euclidean distance를 계산:

$$\text{EDT}(p) = \min_{q \in \text{Background}} \|p - q\|_2$$

이 값에 2를 곱하면 해당 위치의 혈관 직경(두께)을 추정할 수 있다:

$$\text{thickness}(p) = 2 \times \text{EDT}(p)$$

직관: EDT는 혈관 중심에서 최대값을 가지며, 이는 해당 위치의 반지름에 해당. ×2로 직경을 얻음.

#### Skeletonization과 Node/Edge 형성

`skimage.morphology.skeletonize`로 혈관 마스크를 1-pixel 너비의 중심선으로 축소한 뒤:

**노드 검출**: 각 skeleton 픽셀의 3×3 이웃에서 8-connected neighbor 수를 계산.

| Neighbor 수 | 분류 | 의미 |
|-------------|------|------|
| 1 | Endpoint | 혈관 말단 |
| 2 | 경로 픽셀 | 노드 아님 (edge의 일부) |
| 3+ | Junction | 혈관 분기점 |

**Edge 형성**: 각 노드에서 출발하여 skeleton 픽셀을 따라 다음 노드까지 walk. 경로를 따라 다음 속성을 계산:

| 속성 | 계산 방법 |
|------|-----------|
| `weight` | 경로 길이 (픽셀 수) |
| `avg_thickness` | 경로 픽셀들의 thickness map 평균 |
| `vessel_type` | 경로 픽셀들의 AV mask 다수결 투표 (artery/vein/unknown) |

**AV Mask 색상 인코딩**: Red (255,0,0) = Artery, Blue (0,0,255) = Vein

#### 거시적 그래프 통계 추출 (`extract_macro_features.py`)

개별 edge/node 수준이 아닌, **그래프 전체 수준**의 통계치를 추출하여 혈관 네트워크의 전반적 특성을 정량화:

| 특징 | 계산 | 임상적 의미 |
|------|------|-------------|
| `vessel_density` | 혈관 픽셀 수 / 전체 픽셀 수 (바이너리 마스크) | 전체 혈관 면적 비율. 질환군에서 혈관 희소화 반영 |
| `avg_edge_thickness` | 전체 edge의 `avg_thickness` 산술 평균 | 혈관의 전반적 굵기. 질환에 의한 혈관 협착 감지 |
| `total_junctions` | `node_type == "junction"` 노드 수 | 혈관 분기점 수. 모세혈관 소실 척도 |
| `total_endpoints` | `node_type == "endpoint"` 노드 수 | 혈관 말단 수. 네트워크 연결성 지표 |
| `total_edges` | 그래프 edge 총 수 | 혈관 세그먼트 수. 혈관 복잡도 |
| `avg_edge_length` | 전체 edge의 `weight` 산술 평균 | 혈관 세그먼트 평균 길이 |

이 거시 통계의 핵심 역할: TDA 특징(persistent homology)은 혈관 네트워크의 **위상학적 구조**(연결 성분의 birth/death 패턴)를 포착하지만, "혈관이 전체적으로 얼마나 많은가, 얼마나 두꺼운가"라는 **형태학적 양** 자체는 반영하지 못한다. 거시 그래프 통계는 이 gap을 메워 Stage 1(Normal vs Disease) 분리에 필수적인 정보를 제공.

**입출력**
- 입력: `01_VCP/results/logs/vessel_png/{id}_vessel.png` + `processed_graphs/{id}_graph.pkl`
- 출력: `02_Graph_Extraction/graph_macro_features.csv`

---

### 3-6. Track A — Radial Filtration TDA (`03_trackA/`)

**배경**: Optic Disc를 중심으로 방사형 필트레이션을 수행하여 녹내장(Glaucoma) 스크리닝을 위한 TDA 특징 추출.

#### 방사형 필트레이션 (Radial Filtration)

OD mask의 center of mass $(y_c, x_c)$를 기준으로, 그래프의 각 노드에 OD로부터의 Euclidean distance를 필트레이션 값으로 할당:

$$f(v) = \sqrt{(v_y - y_c)^2 + (v_x - x_c)^2}$$

각 Edge에는 simplex property를 유지하기 위해:

$$f(e) = \max(f(u), f(v))$$

이는 "OD에서 바깥으로 원이 퍼져나가면서 혈관 그래프를 점진적으로 드러내는" 과정을 모델링. 필트레이션 값이 증가함에 따라 OD 근처의 노드가 먼저 나타나고, 말초 혈관이 나중에 나타남.

#### 위상학적 해석 — Persistent Homology

GUDHI `SimplexTree`에 모든 vertex와 edge를 삽입한 뒤 `compute_persistence()`를 수행:

- **$\beta_0$ (H0, 연결 성분)**: 필트레이션이 진행되면서 새 노드가 나타날 때 새 연결 성분이 생기고(birth), edge가 두 성분을 연결하면 하나가 사라짐(death). **Lifespan = death - birth**는 두 혈관 가지가 얼마나 먼 거리에서 합류하는지를 나타냄. 녹내장에서는 시신경 손상으로 혈관 가지가 줄어들어 이 패턴이 변화.

- **$\beta_1$ (H1, 루프)**: 망막 혈관은 대부분 tree 구조(비순환)이므로 H1은 일반적으로 0. 루프가 존재하면 혈관 문합(anastomosis)을 의미.

**추출 특징**

| 특징 | 차원 | 설명 |
|------|------|------|
| `b0_max_lifespan` | H0 | 가장 오래 지속된 연결 성분의 수명 |
| `b0_sum_lifespan` | H0 | 모든 연결 성분 수명의 총합 |
| `b1_count` | H1 | noise threshold(5px)를 넘는 루프 수 |
| `b1_max_lifespan` | H1 | 가장 오래 지속된 루프의 수명 |
| `persistence_entropy` | H1 | H1 persistence diagram의 Shannon 엔트로피 |

**입출력**
- 입력: `processed_graphs/{id}_graph.pkl` + `od_png/{id}_od.png`
- 출력: `03_trackA/trackA_features.csv`

---

### 3-7. Track B — 알츠하이머 바이오마커 (`03_trackB/`)

**배경**: 알츠하이머 질환(AD)에서 관찰되는 망막 혈관 변화를 세 가지 바이오마커로 정량화.

#### Box-counting Fractal Dimension

혈관 마스크의 구조적 복잡도를 프랙탈 차원으로 측정.

**알고리즘**: 다양한 크기 $s$의 정사각형 box로 이미지를 덮고, 혈관 픽셀을 포함하는 box 수 $N(s)$를 세어:

$$D = \lim_{s \to 0} \frac{\log N(s)}{\log(1/s)}$$

실제로는 $s \in \{2, 4, 8, 16, 32, 64, 128\}$에 대해 $\log(N(s))$ vs $\log(1/s)$를 선형 회귀하여 기울기를 $D$로 사용.

- 직선: $D \approx 1.0$
- 완전히 채워진 평면: $D \approx 2.0$
- 정상 망막 혈관: $D \approx 1.3 \sim 1.55$
- AD 환자는 혈관 희소화로 $D$가 감소하는 경향

**수학적 의미**: Box-counting dimension은 Hausdorff 차원의 상한(upper bound)으로, 자기 유사적(self-similar) 프랙탈 구조의 복잡도를 단일 스칼라로 요약한다. 혈관 네트워크의 분기 밀도와 공간 충전율을 동시에 반영하므로, 신경퇴행으로 인한 혈관 희소화(rarefaction)를 감지하는 민감한 지표로 기능.

#### Artery-Vein Ratio (AVR)

그래프 edge의 `avg_thickness`와 `vessel_type` 속성을 이용:

$$\text{AVR} = \frac{\bar{T}_\text{artery}}{\bar{T}_\text{vein}}$$

여기서 $\bar{T}_\text{artery}$는 artery로 분류된 모든 edge의 평균 두께, $\bar{T}_\text{vein}$은 vein edge의 평균 두께. 동맥 또는 정맥이 없으면 AVR = 0으로 처리.

정상인은 AVR ≈ 0.6~0.8. AD 환자에서는 동맥 협착으로 AVR이 감소하는 경향.

#### Flooding Filtration

혈관 두께 기반 "홍수" 시뮬레이션:

- 노드 필트레이션: $f(v) = -\max_{e \ni v} \text{avg\_thickness}(e)$ (부호 반전으로 sublevel 필트레이션)
- 엣지 필트레이션: $f(e) = \max(f(u), f(v))$

두꺼운 주요 혈관이 먼저 나타나고, 가는 말초 혈관이 나중에 드러남. Track A의 방사형 필트레이션과는 다른 관점(공간적 거리 vs 혈관 두께)으로 구조를 분석.

**Track A vs Track B 필트레이션 비교**:

| 관점 | Track A (Radial) | Track B (Flooding) |
|------|------------------|--------------------|
| 필트레이션 기준 | OD로부터의 공간적 거리 | 혈관 두께 (형태학적) |
| 직관 | OD에서 원이 퍼져나감 | 두꺼운 혈관부터 물이 차오름 |
| 포착하는 구조 | 혈관 분기의 공간 배치 | 혈관 계층 구조 (주혈관 → 세혈관) |
| 대상 질환 | 녹내장 (시신경 주변 혈관 변화) | AD (전체 혈관 밀도/두께 변화) |

**추출 특징**: `flood_b0_max_lifespan`, `flood_b0_sum_lifespan`, `flood_b1_count`, `flood_b1_max_lifespan`, `flood_persistence_entropy`

**입출력**
- 입력: `processed_graphs/{id}_graph.pkl` + `vessel_png/{id}_vessel.png`
- 출력: `03_trackB/trackB_features.csv`

---

### 3-8. EDA & Feature Selection (`04_EDA_and_Selection/`)

**배경**: 추출된 18개 특징(TDA 12 + Graph Macro 6) 중 통계적으로 유의미한 것만 선별하여 분류기에 전달. 특히, 계층적 분류기의 각 단계에 최적화된 특징을 찾기 위해 **3단계 통계 검정**을 수행.

#### ML 이전에 통계 검정을 수행하는 이유

DRIVE(20) + IOSTAR(30) = 총 50개의 소규모 샘플에서 18개 특징을 모두 사용하면:

1. **차원의 저주 (Curse of Dimensionality)** — 샘플 수 대비 특징 수가 너무 많으면 SVM이 학습 데이터에 과적합. 경험적으로 $n > 5d$ (샘플 수 > 5 × 특징 수)를 권장하는데, $50 / 18 \approx 2.8$로 이 기준을 크게 미달. 실제로 6개 특징을 Stage 1에 투입한 실험에서 Accuracy가 64.0% → 55.3%로 하락하여 차원의 저주를 실험적으로 확인.

2. **노이즈 특징 (Zero-variance Features)** — 망막 혈관의 tree 구조 특성상 모든 H1 특징(루프 관련)이 분산 0으로 수렴. 이러한 상수 특징은 분류에 기여하지 않고 모델 복잡도만 증가시킴.

3. **다중공선성** — 상관관계가 높은 특징 쌍은 SVM의 결정 경계를 불안정하게 만들고, 하이퍼파라미터 튜닝 결과의 재현성을 떨어뜨림.

4. **해석 가능성** — 의료 분류에서는 "왜 이 진단이 내려졌는가"를 설명할 수 있어야 함. 소수의 특징이면 결정 경계를 시각화하여 임상적 해석이 가능.

#### 3단계 통계 검정 — Stage-Specific Feature Selection의 필요성

초기 접근법에서는 3-class ANOVA/Kruskal-Wallis 검정만 수행하여 전체적으로 유의한 특징 2개(`fractal_dimension`, `b0_sum_lifespan`)를 선별했다. 이 특징들은 AD vs Glaucoma 구분(Stage 2)에서 p < 0.01로 매우 강력했지만, Normal vs Disease 구분(Stage 1)에서는 **어떤 TDA 특징도 p < 0.05에 도달하지 못하는** 근본적 한계가 확인되었다.

이는 다음과 같은 통찰을 제공한다:

- **3-class 검정에서 유의한 특징 ≠ 모든 이진 분류에 유의한 특징**. 3-class에서 유의하다는 것은 "세 그룹 중 최소 하나가 다르다"는 의미이지, 모든 쌍별 비교에서 유의하다는 의미가 아님.
- Normal vs Disease의 차이는 **거시적(혈관 밀도, 두께)** 수준에서 나타나며, TDA가 포착하는 **미시적(위상학적 구조)** 변화와는 다른 스케일의 현상.

따라서 계층적 분류기의 각 단계에 최적화된 특징을 찾기 위해, 3-class 검정 외에 **2-class binary 검정**을 추가로 수행:

**검정 절차** (각 특징에 대해):

1. **정규성 검정**: Shapiro-Wilk test (각 그룹별)
   - 귀무가설: "데이터가 정규분포를 따른다"
2. **그룹 간 차이 검정**:
   - 3-class: 모든 그룹 정규 → ANOVA, 비정규 → Kruskal-Wallis
   - 2-class: 양 그룹 정규 → Welch t-test, 비정규 → Mann-Whitney U
3. **유의수준**: $p < 0.05$인 특징만 선별

#### 3-class 검정 결과 (Normal / Alzheimer / Glaucoma)

| 특징 | 검정 방법 | p-value | 유의 |
|------|-----------|---------|------|
| **avg_edge_thickness** | Kruskal-Wallis | **0.000077** | YES |
| **vessel_density** | ANOVA | **0.000241** | YES |
| **fractal_dimension** | ANOVA | **0.000413** | YES |
| **total_endpoints** | Kruskal-Wallis | **0.000579** | YES |
| **total_junctions** | Kruskal-Wallis | **0.000800** | YES |
| **total_edges** | Kruskal-Wallis | **0.000861** | YES |
| **b0_sum_lifespan** | Kruskal-Wallis | **0.016513** | YES |
| avg_edge_length | ANOVA | 0.1042 | NO |
| b0_max_lifespan | Kruskal-Wallis | 0.1245 | NO |
| AVR | Kruskal-Wallis | 0.3018 | NO |
| H1 특징 (6개) | — | — | 분산 0 |

#### Stage 1 검정 결과 (Normal vs Disease)

| 특징 | 검정 방법 | p-value | 유의 |
|------|-----------|---------|------|
| **avg_edge_thickness** | Mann-Whitney U | **0.0228** | **YES** |
| vessel_density | Welch t-test | 0.0513 | 경계선 |
| fractal_dimension | Mann-Whitney U | 0.0733 | NO |
| b0_sum_lifespan | Mann-Whitney U | 0.0941 | NO |

**핵심 발견**: 3-class에서 가장 유의했던 TDA 특징들이 Stage 1에서는 유의하지 않으며, 거시적 그래프 특징인 `avg_edge_thickness`만이 유일하게 p < 0.05를 달성. 이는 정상군과 질환군의 차이가 위상학적 구조가 아닌 **혈관의 전반적 형태학적 변화**(두께 감소, 밀도 감소)에 있음을 시사.

#### Stage 2 검정 결과 (Alzheimer vs Glaucoma)

| 특징 | 검정 방법 | p-value | 유의 |
|------|-----------|---------|------|
| **total_endpoints** | Welch t-test | **< 0.0001** | YES |
| **total_junctions** | Welch t-test | **< 0.0001** | YES |
| **total_edges** | Welch t-test | **< 0.0001** | YES |
| **avg_edge_thickness** | Welch t-test | **0.000033** | YES |
| **vessel_density** | Welch t-test | **0.001719** | YES |
| **fractal_dimension** | Welch t-test | **0.007352** | YES |
| **b0_sum_lifespan** | Mann-Whitney U | **0.013455** | YES |
| **avg_edge_length** | Welch t-test | **0.022069** | YES |

**핵심 발견**: AD와 Glaucoma 사이에서는 거시적 그래프 특징과 TDA 특징 **모두** 극도로 유의. 두 질환의 망막 혈관 변화 패턴이 근본적으로 다름을 증명.

#### Stage-Specific 검정이 규명한 "두 스케일의 생물학적 차이"

```
정상 vs 질환 (Stage 1):
  → 거시적 변화: 혈관이 전반적으로 얇아지고, 밀도가 낮아짐
  → 그래프 수준 통계(avg_edge_thickness, vessel_density)로 포착
  → TDA 특징은 이 "양적 변화"에 둔감

AD vs Glaucoma (Stage 2):
  → 미시적 변화: 위상학적 붕괴 패턴이 질적으로 다름
  → AD: 전반적 혈관 복잡도 감소 (fractal dimension ↓)
  → Glaucoma: OD 주변 국소적 혈관 가지 소실 (b0_sum_lifespan ↓)
  → TDA 특징이 이 "질적 차이"를 정밀하게 포착
```

**출력**
- `statistical_results.csv`, `statistical_results_stage1.csv`, `statistical_results_stage2.csv`
- `selected_features.csv`, `selected_features_stage1.csv`, `selected_features_stage2.csv`
- `plots/boxplots.png`, `plots/correlation_heatmap.png`

---

### 3-9. Feature-Specific Hierarchical SVM (`05_SVM/`)

**배경**: Normal(23) / Alzheimer(20) / Glaucoma(7)의 3-class 분류를 수행. 소규모 불균형 데이터(N=50)에서 과적합을 방지하고, Stage-specific EDA에서 발견된 "두 스케일"의 차이를 최대한 활용하기 위해 **Feature-Specific Hierarchical Architecture**를 채택.

#### 왜 계층적 분류 + 특징 슬라이싱이 필요한가

단순 3-class flat SVM은 41.3%에 그쳤다. 문제의 근원:

1. **극심한 클래스 불균형** — Glaucoma(7)가 Normal(23)의 1/3. 소수 클래스가 결정 경계에 미치는 영향이 미약.
2. **이질적 분리 스케일** — Normal/Disease 경계는 거시적, AD/Glaucoma 경계는 미시적. 하나의 특징 공간에서 두 경계를 동시에 학습하는 것은 N=50에서 비현실적.
3. **차원의 저주** — 18개 특징 전부를 투입하면 $n/d = 2.8$로 과적합이 필연적.

**해법: 2단계 이진 분류 + 단계별 특징 최적화**

```
Stage 1: Normal(23) vs Disease(27)
  → 특징: avg_edge_thickness, vessel_density, fractal_dimension, b0_sum_lifespan
  → 거시적 그래프 통계 + TDA 보강 (4개, p < 0.1)
  → 거의 균형 데이터 (23 vs 27)

Stage 2: Alzheimer(20) vs Glaucoma(7) — Disease로 분류된 샘플만
  → 특징: fractal_dimension, b0_sum_lifespan
  → 순수 TDA 특징 (2개, p < 0.01)
  → class_weight='balanced'로 불균형 보상
```

| 설계 결정 | 근거 |
|-----------|------|
| Stage 1에 그래프 통계 투입 | Stage 1 binary test에서 유일하게 p < 0.05인 `avg_edge_thickness` 활용 |
| Stage 2에 TDA만 투입 | Stage 2에서 TDA 2개가 이미 83.3% 달성; 추가 특징은 노이즈만 증가 |
| Stage 1에 TDA도 병용 | 그래프 통계 단독(62.7%) < 그래프+TDA 합산(64.7%). 보완적 역할 확인 |
| 각 단계 2~4개 특징 | $n/d$ 비율 유지로 차원의 저주 방지 |

#### Pipeline 기반 학습 — 데이터 누출 방지

SVM은 특징의 스케일에 민감하므로 `StandardScaler`로 정규화가 필수. 이때 흔한 실수는 전체 데이터에 scaler를 먼저 fit한 뒤 CV를 수행하는 것인데, 이는 테스트 fold의 통계가 학습 fold에 유출되는 **데이터 누출(data leakage)** 을 야기한다.

이를 방지하기 위해 `sklearn.pipeline.Pipeline`으로 `StandardScaler`와 `SVC`를 결합:

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(probability=True, class_weight="balanced")),
])
```

Pipeline 내부에서 `scaler.fit_transform()`은 학습 fold에서만 실행되고, 테스트 fold에는 `scaler.transform()`만 적용. GridSearchCV에 이 Pipeline을 전달하면 내부 CV에서도 동일한 보호가 적용됨.

**데이터 누출이 문제인 이유**: 전체 데이터에 scaler를 fit하면 테스트 데이터의 평균/표준편차가 학습 과정에 반영됨. N=50에서는 테스트 fold (10개 샘플)의 통계가 전체 통계에 상당한 영향을 미쳐, 성능이 실제보다 0.05~0.15 과대추정될 수 있음.

#### Repeated Stratified K-Fold Cross-Validation

총 50개 샘플에서 단순 train/test split은:

- 학습 데이터 부족으로 모델이 불안정
- 한 번의 split에 따라 결과가 크게 변동

**단일 Stratified K-Fold** (K=5)도 N=50에서는 fold 구성에 따른 분산이 크다. 이를 해결하기 위해 **Repeated Stratified K-Fold** (3회 반복 × 5-fold = 15회 평가)를 적용:

1. 각 반복(repeat)에서 다른 랜덤 시드로 데이터를 5개 fold로 재분할
2. 클래스 비율은 모든 fold에서 유지 (Stratified)
3. 15번의 결과를 평균하고 **표준편차**를 함께 보고하여 성능 추정의 불확실성을 정량화

이 방식의 장점:
- 단일 K-Fold 대비 분산이 $\frac{1}{\sqrt{N\_REPEATS}} \approx 0.577$배로 감소
- 특정 fold 구성에 의한 운(luck)의 영향을 완화
- mean ± std 보고로 성능의 신뢰 구간 제공

내부(inner) CV도 Stratified으로 수행하되, 가장 작은 클래스(Glaucoma, N=7)의 학습 fold 내 샘플 수에 따라 inner fold 수를 동적으로 조절:

$$k_\text{inner} = \max(2, \min(3, n_\text{min\_class\_in\_train}))$$

이는 inner CV에서 fold당 최소 1개 이상의 Glaucoma 샘플이 보장되도록 함.

#### 클래스 불균형 처리

데이터 분포가 Normal(23) / Alzheimer(20) / Glaucoma(7)로 불균형:

**`class_weight='balanced'` 선택 이유**:

- 각 클래스의 가중치를 $w_c = \frac{N}{k \times n_c}$로 설정
  - $N$: 전체 샘플 수, $k$: 클래스 수, $n_c$: 해당 클래스 샘플 수
  - 실제 가중치: Normal ≈ 0.72, Alzheimer ≈ 0.83, **Glaucoma ≈ 2.38**
  - Glaucoma의 오분류 비용이 Normal의 ~3.3배로 자동 설정됨

**SMOTE를 사용하지 않는 이유**:

1. N=50에서 SMOTE는 기존 소수 클래스(Glaucoma 7개)의 k-NN 보간으로 합성 샘플을 생성하는데, 7개의 실제 데이터 포인트에서 생성된 합성 포인트는 기존 분포를 충실히 반영하지 못하고 노이즈 패턴까지 증폭
2. 2차원 특징 공간에서 7개 포인트의 볼록 껍질(convex hull) 내부에만 합성 포인트가 생성되므로, 실제 분포가 이 영역 밖에도 존재할 수 있어 편향 유발
3. 실험적 검증: SMOTE(k=2) 적용 시 accuracy가 41.3% → 39.3%로 오히려 하락
4. `class_weight`는 손실 함수 수준에서 보상하므로 데이터 분포 자체를 왜곡하지 않음

#### 계층적 예측 + 확률 전파

테스트 시, Stage 1의 확률 추정이 Stage 2로 전파되어 최종 3-class 확률을 구성:

```python
# Stage 1: P(Disease), P(Normal)
# Stage 2 (Disease로 분류된 샘플만): P(AD|Disease), P(Glaucoma|Disease)

# 최종 확률:
P(AD)       = P(Disease) × P(AD|Disease)
P(Glaucoma) = P(Disease) × P(Glaucoma|Disease)
P(Normal)   = P(Normal)   # from Stage 1
```

이 확률 전파 방식은 Stage 1의 불확실성이 Stage 2의 최종 확률에 반영되므로, ROC/AUROC 계산이 계층 구조를 정확히 반영.

#### GridSearchCV 하이퍼파라미터 튜닝

탐색 공간 (Stage 1, Stage 2 동일):

| 파라미터 | 후보값 |
|----------|--------|
| `kernel` | `linear`, `rbf` |
| `C` | 0.01, 0.1, 1, 10, 100 |
| `gamma` (rbf만) | `scale`, `auto`, 0.1, 0.01, 0.001 |

평가 기준: `f1_macro` (불균형 데이터에서 accuracy보다 공정한 지표)

#### 평가 지표

| 지표 | 설명 | 계산 |
|------|------|------|
| Accuracy | 전체 정확도 | $\frac{\text{TP}+\text{TN}}{\text{Total}}$ |
| Sensitivity (Recall) | 양성 검출률 (macro) | $\frac{\text{TP}}{\text{TP}+\text{FN}}$ per class, 평균 |
| Specificity | 음성 배제률 (macro) | $\frac{\text{TN}}{\text{TN}+\text{FP}}$ per class, 평균 |
| Precision | 양성 예측도 (macro) | $\frac{\text{TP}}{\text{TP}+\text{FP}}$ per class, 평균 |
| F1-Score | 정밀도-재현율 조화평균 (macro) | $2 \times \frac{P \times R}{P + R}$ per class, 평균 |
| AUROC | ROC 곡선 아래 면적 (OVR macro) | One-vs-Rest 이진화 → per-class AUC → 평균 |

추가로 **Stage별 개별 지표** (s1_acc, s1_f1, s2_acc, s2_f1)를 추적하여 병목 단계를 식별.

#### 실험 이력 및 아키텍처 진화

| 버전 | 아키텍처 | Stage 1 Acc | Stage 2 Acc | 3-class Acc |
|------|----------|-------------|-------------|-------------|
| v1 | Flat 3-class SVM (TDA 2f) | — | — | 41.3% |
| v2 | + SMOTE (k=2) | — | — | 39.3% |
| v3 | Hierarchical (TDA 2f 공유) | 64.0% | 83.3% | 58.7% |
| v4 | + TDA 6f Stage 1 only | 55.3% | 83.3% | 50.0% |
| v5 | + Graph Macro 2f Stage 1 | 62.7% | 83.3% | 60.0% |
| **v6** | **Graph+TDA 4f Stage 1** | **64.7%** | **83.3%** | **60.7%** |

v4에서 TDA 6개 특징 투입 시 성능이 하락한 것은 N=50에서의 차원의 저주를 실험적으로 증명. v5→v6에서 그래프 거시 통계와 TDA 특징의 **보완적 결합**이 최적임을 확인.

#### 과적합 방지 전략 요약

| 전략 | 구현 | 효과 |
|------|------|------|
| Pipeline 기반 scaling | `Pipeline([scaler, svc])` | 데이터 누출 방지 |
| Repeated Stratified K-Fold | 3×5 = 15 evaluations | 성능 추정 분산 감소 |
| 사전 특징 선별 | 18 → 4+2 features (stage별) | 차원의 저주 완화 |
| class_weight='balanced' | 손실 함수 가중치 조정 | 소수 클래스 보상 |
| Hierarchical decomposition | 3-class → 2 binary | 각 단계의 학습 복잡도 감소 |
| Stage-specific feature slicing | Stage 1: 4f, Stage 2: 2f | 각 단계에 최적화된 정보만 투입 |
| GridSearchCV (f1_macro) | 내부 CV로 하이퍼파라미터 선택 | 과적합 하이퍼파라미터 방지 |
| Inner CV fold 동적 조절 | $k = \max(2, \min(3, n_\min))$ | StratifiedKFold 안정성 확보 |

#### 출력

- `results/confusion_matrix.png`: 행 정규화 혼동 행렬 (비율 + 원시 카운트 표시)
- `results/roc_curves.png`: 클래스별 OVR ROC 곡선 + macro AUC
- `results/decision_boundary.png`: Stage 1 (4D → top-2 scatter) + Stage 2 (2D 결정 경계)
- `results/fold_metrics.csv`: repeat/fold별 3-class + stage별 Acc/F1 + mean ± std 행
- `results/best_svm_model.pkl`: Stage1 Pipeline + Stage2 Pipeline + LabelEncoder + 특징 이름 (배포용)

---

## 4. 모델 구조 상세 (01_VCP)

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

## 5. 설정값 (01_VCP/config.py)

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

## 6. 전체 출력 경로

```
01_VCP/results/logs/
├── vessel_png/            혈관 마스크
├── pixelwise_png/         AV 분류 (pixelwise)
├── od_png/                OD 분할
├── topology_png/          토폴로지 시각화
├── thickness_png/         두께 맵
└── orientation_png/       방향 맵

02_Graph_Extraction/
├── processed_graphs/
│   ├── DRIVE/             {id}_graph.pkl (20개)
│   └── IOSTAR/            {id}_graph.pkl (30개)
└── graph_macro_features.csv  거시적 그래프 통계 (50행 × 8열)

03_trackA/trackA_features.csv      방사형 TDA 특징 (50행 × 7열)
03_trackB/trackB_features.csv      AD 바이오마커 (50행 × 9열)

04_EDA_and_Selection/
├── statistical_results.csv        3-class 통계 검정 결과
├── statistical_results_stage1.csv Stage 1 검정 결과 (Normal vs Disease)
├── statistical_results_stage2.csv Stage 2 검정 결과 (AD vs Glaucoma)
├── selected_features.csv          3-class 선별 특징 (7개)
├── selected_features_stage1.csv   Stage 1 선별 특징
├── selected_features_stage2.csv   Stage 2 선별 특징
└── plots/                         학술 시각화

05_SVM/results/
├── confusion_matrix.png           혼동 행렬
├── roc_curves.png                 OVR ROC 곡선
├── decision_boundary.png          Stage 1 + Stage 2 결정 경계
├── fold_metrics.csv               fold별 + stage별 성능 표 (mean ± std)
└── best_svm_model.pkl             Hierarchical Pipeline (Stage1 + Stage2)
```

---

## 7. 전체 파이프라인 실행 순서

```bash
# 1. VCP — 전처리, 학습, 추론
cd 01_VCP
conda activate vcp
bash 01_preprocess.sh && bash 02_train.sh && bash 03_inference.sh

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

# 6. SVM 분류기 — Feature-Specific Hierarchical SVM
cd ../05_SVM
python svm_classifier.py
```
