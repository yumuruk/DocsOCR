# 평가 프로토콜

Phase 1 평가 하네스가 무엇을 어떻게 재는지 정의하는 문서. 숫자를 만드는 규칙은 여기서만 바꾼다.
실측 결과는 [EXPERIMENTS.md](EXPERIMENTS.md)(원장), 판단과 경위는 [RESEARCH_LOG.md](RESEARCH_LOG.md)(서술)에 있다.
이 하네스가 왜 필요했는지는 [../FINDINGS.md](../FINDINGS.md) Phase 0 실사를 참조.

## 0. 이 프로젝트가 검증하려는 것

기존 방법론(전체 페이지 OCR, 범용 사전학습 레이아웃 모델, 고정 규칙)의 한계는 두 가지다.

- **비용**: 헤더·꼬리말·로고·figure까지 전부 OCR/LLM에 넣어 연산과 토큰을 낭비한다.
- **품질**: 반복되는 문서 furniture가 출력에 섞여 다운스트림 입력을 오염시킨다.

개선 가설은 *문서군당 소량(10~20페이지) 어노테이션으로 레이아웃 모델을 적응시키면,
라벨별 정책으로 OCR을 선별 실행해 비용과 노이즈를 동시에 줄일 수 있다*는 것이다.

이 가설은 **난이도가 다른 두 과제**로 쪼개진다. 하네스가 지표를 둘로 나누는 이유다.

| 과제 | 성격 | 일반화 | 적합한 지표 |
|---|---|---|---|
| **스킵** (header/footer/figure/ignore) | 거친 판단. 위치·텍스트 밀도·종횡비 같은 저수준 단서로 결정되어 레이아웃이 달라도 전이된다 | 강함 | 면적 기반 운영 지표 (§2) |
| **추출** (table/target_data) | 세밀한 판단. "이 문서군에서 이 필드가 여기 있다"는 고수준 관습에 의존 | 약함 | per-class AP (§3) |

단일 mAP는 이 둘을 평균으로 뭉갠다. 그래서 리포트는 운영 지표를 먼저 싣는다.

## 1. 측정 전제

| 항목 | 값 | 근거 |
|---|---|---|
| 정본 라벨 | `labels.CANONICAL_LABELS` 8종 고정 순서 | 클래스 id가 파일에 종속되면 데이터셋 간 비교가 무의미 (FINDINGS C3) |
| 예측 생성 conf | **0.001** (`EVAL_CONFIDENCE`) | 0.25로 예측하면 PR 곡선 recall 꼬리가 잘려 AP가 체계적으로 과소평가 (C4) |
| 운영 작동점 conf | **0.25** (`OPERATING_CONFIDENCE`) | 운영 지표·혼동행렬은 실제 배포 동작을 서술해야 함 |
| IoU | `evaluation.geometry.iou` | `BoundingBox.intersects()`는 IoU가 아니라 `intersection/min(area)` — 재사용 금지 |
| 분할 | 문서 단위, 시드 고정, test 분리 | 같은 문서의 페이지는 근사 중복이므로 페이지 단위 분할은 누수 (C1) |
| 빈 페이지 | `label_status == complete`만 채택 | 미완성 페이지가 "아무것도 없는 페이지"로 들어가는 것을 차단 (C5) |

**라벨 매핑.** 사전학습 DocStructBench 체크포인트는 정본 라벨을 모른다. zero-shot 평가에는
`--label-mapping docstructbench`가 필수이고, 매핑은 `labels.DOCSTRUCTBENCH_TO_CANONICAL`에 명시돼 있다.
파인튜닝한 모델은 정본 라벨을 직접 학습하므로 `identity`를 쓴다. 매핑을 잘못 고르면 결과가 조용히 무의미해지므로
사용된 매핑 이름은 모든 리포트에 기록된다.

`target_data`는 DocStructBench에 대응 클래스가 없다. **zero-shot에서 이 클래스 AP 0은 모델 실패가 아니라
라벨 공간의 구조적 결과**이고, 리포트가 이를 자동으로 명시한다.

## 2. 운영 지표 — 선별 OCR vs 전체 페이지 OCR

`pred_keep` = 작동점(0.25) 이상 + keep 라벨인 예측 박스의 **기하학적 합집합**.
겹치는 박스는 한 번만 센다(중복 과금 방지). keep/skip은 영역 정책에서 파생한다:

- keep: `title`, `body`, `table`, `target_data`
- skip: `figure`, `header`, `footer`, `ignore`

| 지표 | 정의 | 방향 |
|---|---|---|
| **절감률** reduction | `1 - area(pred_keep) / area(page)` | 높을수록 좋음 |
| **유실률** loss | `area(gt_keep \ pred_keep) / area(gt_keep)` | **하드 제약** |
| **오염률** contamination | `area(pred_keep \ gt_keep) / area(pred_keep)` | 낮을수록 좋음 |

집계는 면적 가중(페이지 면적 합으로 나눔)이다. 페이지별 비율의 단순 평균은 거의 빈 페이지와
빽빽한 페이지에 같은 가중을 주므로 쓰지 않는다.

**목적함수는 평균이 아니다.** 유실은 내용이 사라지는 것이고 절감은 돈이므로 교환 대상이 아니다.
운영 규칙: **유실률 상한을 먼저 걸고(예: ≤ 1%), 그 제약 아래 절감률을 최대화**한다.
mAP가 높은데 유실률이 큰 셀은 탈락이다.

### 기준선 2개를 같은 표에 넣는다

| 기준선 | 의미 |
|---|---|
| `full_page_ocr` | 기존 방법론. 절감 0, 유실 0, **오염률이 제거 대상 노이즈의 크기** |
| `oracle` | 예측을 GT로 대체. **절감률 상한**. 모델과의 차이가 남은 개선 여지 |

항등식 하나가 성립하며 테스트로 고정돼 있다: `oracle` 절감률 == `full_page_ocr` 오염률
(둘 다 `1 - gt_keep/page`). 어긋나면 면적 계산이 틀린 것이다.

## 3. 검출 지표 — per-class AP

COCO 관례를 따른다. `pycocotools`와 대조 검증돼 있다(`tests/test_evaluation.py::PycocotoolsParityTests`,
AP@0.5 / 0.50:0.95 전 구간 및 다중 클래스·중복 검출 케이스).

- 점수 내림차순 greedy 1:1 매칭, 같은 클래스·같은 페이지의 최고 IoU 미점유 GT에 배정
- precision envelope을 단조 감소로 만든 뒤 101개 recall 지점 평균
- AP@0.5:0.95 = IoU 0.50~0.95 (0.05 간격) 평균
- **GT가 0인 클래스는 `None`이고 mAP 평균에서 제외**한다. "예시가 없음"과 "다 틀렸음"은 다르다

**혼동행렬**은 클래스 무관 매칭(IoU≥0.5, 작동점 conf)으로 만든다. table을 figure로 본 것이
"miss 1 + false positive 1"이 아니라 혼동 1건으로 보이게 하려는 것이다. `table↔figure`,
`header↔title`이 주 관찰 대상이다.

## 4. 실험 축

> **현행 실행 계획**: 목표가 RAG 코퍼스 정제(범용 내용/잡동사니 분리)로 확정되면서
> 실행 순서는 **Phase A(zero-shot 장르 실측, 학습 0회) → Phase B(무너진 장르만
> fine-tuning)** 로 재편됐다 — README의 "실험 설계" 절과 RESEARCH_LOG 2026-08-16 항목
> 참조. 아래 축들은 Phase B에서 유효하며, `init` 축은 DocStructBench 고정으로
> 단순화될 수 있다.

| 축 | 수준 | 상태 |
|---|---|---|
| 라벨 페이지 수 `n` | 5 / 10 / 20 / 50 | `splits.sample_train_pages` (중첩 표집) |
| 초기화·학습 `init` | `zeroshot` / `docstructbench` / `docstructbench-freeze` / `coco` / `scratch` | `--pretrain` / `--freeze N` / `--seed` 노출됨 |
| 라벨 출처 `labels` | `manual` / `pseudo` (무수정 self-training) / `hitl` (예측 후 수정) | `AnnotationSource`로 태깅. `hitl`은 UI 보강 선행 |
| 증강 `mosaic` | `1.0` / `0` | 10페이지에 mosaic 1.0 기본값은 과할 가능성 |
| 시드 `seed` | 0 / 1 / 2 | 작은 n에서 시드 분산이 효과보다 큼 — 반복 필수 |

전부 곱하면 4×5×3×2×3 = 360 run이라 불가능하다. **순차 축소**로 돌린다.

1. **방법 축 선별** — `n=20` 고정, init 5개 × 3시드 = 15 run → 승자 1~2개
2. **라벨 효율 곡선** — 승자만, 4n × 2init × 3시드 = 24 run
3. **라벨 출처 비교** — 최종 승자만, 3출처 × 3시드 = 9 run

각 단계 결과를 보고 다음을 정한다. 1단계에서 "DocStructBench 사전학습이 실제로 이기는가"라는
가장 중요한 질문에 먼저 답이 나온다.

## 5. 라벨링 예산 — 실험 시작 전 필요량

`n=20`만 라벨링하면 실험이 성립하지 않는다. 고정 held-out 테스트셋이 별도로 필요하다.

| 용도 | 페이지 | 조건 |
|---|---|---|
| 고정 테스트셋 | 30~50 | 학습 풀과 **문서 단위로 분리**, 한 번 라벨링 후 동결 |
| 학습 풀 (train) | 50+ | 여기서 n=5/10/20/50을 중첩 표집. `DatasetSplits.train_pool`은 **train만** 반환한다 — val은 학습 중 validation(early stopping, best.pt 선택) 전용이므로 표집 대상이 아니다 |
| val | 별도 | 학습 페이지가 val에 섞이면 학습한 데이터로 모델을 고르게 됨 |
| **합계** | **~90** | 문서 최소 3개 이상 (문서 단위 분할 요건) |

**test 배정은 문서 크기가 서로 다르면 시드와 무관하게 고정된다** (크기 내림차순 배치, 시드는 동일 크기 문서 간 tie-break만 담당). 이는 의도된 동작이다 — 프로토콜의 전제가 "한 번 동결된 test셋"이고, 시드의 역할은 표집·학습 반복이지 held-out의 재정의가 아니다. 문서 크기 편차가 커서 요청 비율을 못 맞추면 분할은 실패하는 대신 `warnings`에 명시한다.

`make_splits`는 문서가 3개 미만이면 거부한다. 단일 문서만 있으면 `strategy="page"`로 강제할 수 있으나
결과에 누수 경고가 붙고 문서 단위 결과와 비교할 수 없다.

## 6. 실행

```bash
# zero-shot 베이스라인 (사전학습 가중치 + 라벨 매핑)
uv run pdf-layout-parser evaluate ./annotations.json ./runs/eval \
  --run-name zeroshot \
  --detector doclayout-yolo \
  --model-path ./models/doclayout_yolo_docstructbench_imgsz1024.pt \
  --label-mapping docstructbench \
  --split test --split-seed 0 \
  --tag init=zeroshot --tag n=0 --tag labels=none

# 파인튜닝 모델 (정본 라벨을 직접 학습하므로 identity 매핑)
uv run pdf-layout-parser evaluate ./annotations.json ./runs/eval \
  --run-name ft-n20-s0 \
  --detector doclayout-yolo \
  --model-path ./runs/train-n20-s0/training/weights/best.pt \
  --label-mapping identity \
  --split test --split-seed 0 \
  --tag init=docstructbench --tag n=20 --tag labels=manual --tag seed=0

# 하한선 대조군
uv run pdf-layout-parser evaluate ./annotations.json ./runs/eval \
  --run-name heuristic --detector heuristic --tag init=heuristic

# 원장에서 셀 비교표 생성
uv run pdf-layout-parser compare ./runs/eval --output docs/comparison.md
```

`evaluate`는 기본으로 [EXPERIMENTS.md](EXPERIMENTS.md)에 행을 **추가**한다(`--no-ledger`로 끌 수 있음).
`--tag key=value`는 반복 가능하고 원장의 `Cell` 열에 실험 좌표로 기록된다.

예측은 `<run-name>_predictions.json`으로 저장되므로, 모델 없이 재채점하려면
`--predictions <path>`로 다시 돌릴 수 있다. 지표 코드를 고친 뒤 과거 run을 전부 재채점할 때 쓴다.

## 7. 기록해야 하는 것

리포트에 자동으로 들어가는 항목(NFR-005 재현성):

- 정본 카탈로그 fingerprint, 사용된 라벨 매핑 이름
- **데이터셋 fingerprint** — **채점된 페이지들**의 GT 내용 해시(전체 코퍼스가 아님).
  같은 값이면 동일 GT로 채점된 것이므로 직접 비교 가능하다. 다르면 비교 불가이며,
  보통 분할 시드가 달라 서로 replicate인 경우다. **행 간 차이를 모델 차이로 읽기 전에 이 열을 먼저 본다.**
- 분할 전략·시드·test 문서 목록
- 예측 conf / 작동점 conf, 채점 소요 시간
- GT 박스 수, 예측 수, 페이지 수

학습 쪽도 자동 기록된다: 모든 학습 run은 `<run_name>_result.json`을 남기고, 여기에는
`YOLOv10.train()`에 **실제로 전달된 kwargs 전체**(seed·freeze 포함), 소요 시간, from-scratch 경고가
들어간다 — preview 문자열도 같은 dict에서 렌더링되므로 기록과 실행이 어긋날 수 없다.
`experiment` 명령은 셀 전체를 `experiments.json` manifest로 저장하며, 모든 셀이 **동일한 val/test를
공유**하고 train만 중첩 표집된다 (`--split-seed` / `--sample-seed` / `--seed`로 각각 분할·표집·학습
시드를 제어).

## 8. 알려진 한계

- **회전 페이지 미처리** — `pdf.py`의 포인트→픽셀 변환이 scale-only. 회전된 페이지의 GT 좌표는 신뢰할 수 없다
- **OCR 정확도는 측정하지 않는다** — 이 하네스는 레이아웃만 본다. `ocr.py`가 언어 미지정(`eng` 기본)이라
  한국어 문서 OCR 품질은 별도 문제이며 이번 범위 밖
- **테이블 구조는 측정하지 않는다** — `table` 영역 검출만 본다. 행/열/셀 정확도는 별도 지표 필요
- **면적 기반 지표는 박스 단위 누락에 둔감** — 작은 `target_data` 박스를 놓쳐도 절감률은 거의 안 변한다.
  그래서 per-class AP를 함께 본다. 두 지표는 서로를 대체하지 않는다
