# FINDINGS — Phase 0 코드 실사 (2026-08-08)

실사 방법: 영역별 감사 6개(core-pipeline / detectors-ocr / dataset-training / cli-api / backend-frontend / tests-docs)를 병렬 수행한 뒤, critical·major 주장 전건에 대해 코드를 다시 읽어 반박을 시도하는 검증 단계를 거쳤다. 아래의 모든 critical·major 항목은 검증을 통과한 것이며, 검증 과정에서 정밀화된 항목은 그 내용을 반영했다.

## 환경 복구 결과

| 항목 | 상태 |
|---|---|
| `uv sync --extra pdf --extra doclayout` | 정상 |
| `frontend` npm install + `npm run build` | 정상 |
| DocStructBench 가중치 (`models/doclayout_yolo_docstructbench_imgsz1024.pt`, 39MB) | 다운로드·로딩 검증 완료. 클래스 10종: `title, plain text, abandon, figure, figure_caption, table, table_caption, table_footnote, isolate_formula, formula_caption` |
| 기존 테스트 (`unittest discover tests`) | 5건 전부 통과 (선택 의존성 없이 순수) |
| 백엔드 기동 (`/health`) | 정상 응답 |

## 브리프의 3대 질문 — 확정 답변

### 1. 평가 코드가 정말 없는가 → **전무하다 (확정)**

`src/`, `backend/`, `tests/` 전체를 grep 전수 조사했다. mAP·IoU·AP·PR·confusion matrix를 계산하는 코드는 단 한 줄도 없다. 유사해 보이는 것들의 정체:

- `training.py:120` `extract_metrics()` — ultralytics `train()` 반환 객체의 `results_dict`를 그대로 담는 수동적 pass-through. 스키마 비보장, 실패 시 `{'repr': ...}`.
- `models.py:44` `BoundingBox.intersects()` — **IoU가 아니다.** intersection / min(두 박스 면적) 비율이다. 텍스트 귀속용 휴리스틱이며, 이를 IoU@0.5 매칭으로 오용하면 작은 GT가 큰 예측에 일부만 겹쳐도 매칭되어 mAP가 크게 부풀려진다. 평가용 IoU는 별도 구현 필수.
- `docs/pdf_parsing_platform_spec.md` FR-010 — 스펙 문서의 약속일 뿐 구현 없음.

### 2. YOLO 변환이 좌표계를 올바르게 다루는가 → **핵심 수식은 정상, 주변부에 결함**

- **정상인 것**: 프론트엔드 export 좌표계와 `dataset.py`의 기대가 정확히 일치함을 확인했다. UI는 SVG `viewBox` + `getScreenCTM().inverse()`로 화면 줌과 무관하게 렌더 이미지(픽셀, dpi 180) 좌표를 export하고(`App.jsx:459`), `to_yolo_bbox()`는 같은 `page.width/height`로 center 정규화한다(`dataset.py:110`). xywh→center 변환 수식 자체도 옳다.
- **결함**:
  - `dataset.py:115` OOB 박스를 center/size **성분별로 독립 clamp** — 이미지 경계를 벗어난 박스가 기하학적으로 틀리게 변환된다 (올바른 방식: x1y1x2y2를 먼저 clamp 후 변환). zero-size 박스도 그대로 기록된다.
  - `dataset.py:33` **클래스 id가 어노테이션 JSON의 `labels` 배열 순서에 종속** — 파일마다 id가 달라질 수 있어 데이터셋 간 모델·라벨 호환이 깨진다. 정본(canonical) 라벨 카탈로그가 코드 어디에도 없다.
  - `dataset.py:54` `labels` 리스트에 없는 라벨의 어노테이션은 **경고 없이 버려진다**.

### 3. `experiment` 서브커맨드의 실제 동작 → **데이터셋 생성 + 선택적 학습뿐, 평가 없음**

- 각 N(`--sample-sizes`)에 대해 `export_yolo_dataset(max_pages=N)` + `run_doclayout_yolo_training()`을 호출한다. 자체 평가 단계는 없다 (`--train` 시 ultralytics 학습 중 내장 validation의 `results_dict`가 pass-through로 남는 게 전부).
- 실행 게이트는 `--dry-run`이 아니라 `--train`이다: `--train` 없으면 항상 dry run (`training.py:158` `dry_run or not run_training`). dry run이어도 데이터셋 생성(이미지 복사 포함)은 실제로 수행된다.
- **치명적**: 샘플링이 "앞에서부터 N페이지" prefix 슬라이스(`dataset.py:29`)이고 각 N마다 80/20 prefix 분할을 다시 하므로 **N마다 val 세트가 다르다** (N=10이면 val=9–10페이지, N=20이면 val=17–20페이지; N=10의 val이 N=20에서는 train에 들어감). 공통 held-out 테스트셋이 존재하지 않아 **이 구조로 그린 라벨 효율 곡선은 점끼리 비교가 불가능하다.**

## Critical — 실험 결과를 조용히 오염시키는 결함

| # | 위치 | 내용 |
|---|---|---|
| C1 | `dataset.py:85-95` | train/val 분할이 셔플·시드 없는 순서 prefix 분할. **문서(document) 개념이 스키마에 없어** 문서 단위 분할이 불가능하고, 페이지 1장이면 `return pages, pages`로 **동일 페이지가 train과 val 양쪽에 들어간다**(완전한 누수). val은 항상 문서 뒷부분 페이지라 순서 편향도 있다. |
| C2 | `dataset.py:29` + `training.py:134-164` | 위 3번 항목: 샘플 크기 실험의 N별 val 불일치 + 공통 테스트셋 부재 → 라벨 효율 곡선 비교 불가. |
| C3 | `dataset.py:33` | 클래스 id가 파일별 `labels` 순서 종속 — 도메인 A/B 데이터셋 간 id가 어긋나면 전이 실험이 조용히 무의미해진다. |
| C4 | `models.py:160`, `detectors.py:87` | 기본 `confidence=0.25`가 **predict 시점에** 적용되어 저신뢰 검출이 아예 반환되지 않는다. 이 경로로 mAP를 계산하면 PR 곡선 recall 꼬리가 잘려 체계적으로 과소평가된다. 평가용 예측 생성 시 conf를 0.001 수준으로 내려야 한다. |
| C5 | `App.jsx:67` + `dataset.py` | 어노테이션이 0개인 페이지도 전부 export되어 **빈 라벨 파일(= 순수 배경 negative 샘플)**이 된다. 부분적으로만 라벨링하고 export하면 "아직 안 그린 페이지"가 "아무것도 없는 페이지"로 학습·평가에 들어가 GT가 조용히 오염된다. |

## Major — 계획된 실험을 막거나 오도하는 결함

- **M1. DocStructBench ↔ 레포 라벨 매핑 부재** (`detectors.py:112`): 어댑터는 모델 클래스명을 `lower()`만 해서 그대로 통과시킨다. 사전학습 가중치의 zero-shot 출력(`plain text`, `abandon`, `figure_caption` 등)과 레포 8라벨(`title/body/table/figure/header/footer/ignore/target_data`) 사이 매핑 테이블이 어디에도 없다. 우연히 겹치는 것은 `title/table/figure/plain text`뿐. 매핑 없이는 zero-shot 베이스라인의 per-class AP·confusion matrix가 성립하지 않는다. (fine-tuning한 모델은 레포 라벨을 그대로 배우므로 무관.) 부작용: `abandon`이 정책 fallback(`pipeline.py:62`) 때문에 skip이 아니라 **OCR 처리**된다.
- **M2. HITL 실험이 현 UI로는 불가능**: UI는 박스 **신규 그리기 + 전체 삭제(Clear)** 만 가능하다(`App.jsx:245`). 개별 박스 삭제·이동·리사이즈·리라벨 불가. 모델 예측은 표시 전용이라 어노테이션으로 프리로드할 수 없고(`App.jsx:104`), 파싱 성공·파일 변경 시 기존 어노테이션이 초기화된다. export된 annotations.json을 다시 불러올 수도 없다. → Phase 4의 "예측을 수정만 하기" 시나리오는 UI 보강 없이는 실험 자체가 불가능. Phase 2의 신규 라벨링도 실수 1번에 전체 삭제라는 리스크를 안고 진행해야 한다.
- **M3. 학습 래퍼의 실험 부적합성** (`training.py`): `pretrain` 기본값 `None` → `yolov10n.yaml` **랜덤 초기화 from-scratch 학습**이다(run_name이 "doclayout_finetune"인 것과 상충; 경로를 직접 주면 파인튜닝 가능). `freeze` 미노출(백본 동결 실험 불가), `seed` 미노출(doclayout_yolo 기본 seed=0이라 기본 재현성은 있으나 replicate 실험 불가, 사용 시드 미기록). 학습 소요 시간·데이터셋 해시 미기록. `ExperimentResult`는 반환만 되고 디스크에 저장되지 않는다(`cli.py`는 stdout 출력뿐).
- **M4. heuristic 베이스라인은 사실상 floor**: 이미지 크기만 읽고 고정 4박스(header 8%/title 10%/body/footer 8%)를 반환한다. `table/figure/target_data`는 절대 출력하지 않으므로 해당 클래스 AP는 항상 0, confidence가 상수라 PR 곡선이 점으로 퇴화한다. "베이스라인 3종 비교"에서 하한선 이상의 의미는 없다.
- **M5. `data.yaml`에 test 스플릿 키가 없다** (`dataset.py:98`): train/val 2-way 구조만 지원. held-out 테스트셋 도입 시 확장 필요.
- **M6. 백엔드는 확인대로 동기 실행이며 더 나쁘다** (`backend/main.py:71`): POST 핸들러 3개가 `async def` 안에서 블로킹 함수(`parse_document`, `export_yolo_dataset`, `run_doclayout_yolo_training`)를 직접 호출 — 스레드풀도 없어 **이벤트 루프 전체가 정지**한다. 실제 학습 요청 1건이 서버를 통째로 얼린다. (이번 범위에서는 실험을 CLI로 돌리면 회피 가능; 개선은 Phase 6.)
- **M7. annotations.json의 `image_path`가 서버 절대경로** (`App.jsx:68`): `runs/web/<uuid>/...` 경로가 그대로 박혀 특정 파싱 런과 파일시스템에 결박된다. 런 디렉터리를 지우면 GT가 죽는다.
- **M8. 테스트 사각지대** (`tests/test_core.py`): YOLO 변환 수치 검증은 인바운드 단일 케이스 1건. clamp·xyxy 변환·분할 누수·좌표 스케일링 전부 미검증. C1의 train==val 중복도 테스트가 잡지 못한다.
- **M9. 스펙 문서와 구현의 전면 불일치** (`docs/pdf_parsing_platform_spec.md`): 스펙의 평가 기능(FR-010)·프로젝트/어노테이션 CRUD·잡 구조 모두 미구현. 스펙은 목표 문서로만 취급할 것.

## Minor / 참고 사실

- `pdf.py:34,78` — 렌더링과 native 텍스트 추출이 각각 독립 `dpi` 인자를 가짐(파이프라인 내부는 일관되나 개별 호출 시 어긋날 수 있음). 포인트→픽셀 변환이 scale-only라 **회전 페이지 미처리**.
- `models.py:115` — result.json에 파싱 조건(dpi·confidence·image_size) 미기록 → 재현성 위해 평가 하네스에서 기록 보강 필요.
- `models.py:24` — bbox 직렬화에 computed `x2/y2` 포함(JSON 6필드). GT/pred 파서가 이를 알아야 함.
- `ocr.py:28` — pytesseract를 언어 미지정(기본 `eng`)으로 호출. 한국어 문서 OCR 품질에 치명적이나 **레이아웃 실험 범위 밖** (OCR 정확도는 이번에 측정하지 않음).
- `App.jsx:89` — UI 경유 파싱은 `device='cpu'` 하드코딩 + `max_pages` 기본 3.
- `training.py:112` — `exist_ok=True` 기본이라 재실행 시 이전 run 덮어쓰기. `find_weight`는 glob 첫 매치 임의 선택.
- confidence는 `DetectedRegion → ParsedRegion → result.json`까지 보존됨 (`raw_label_id`도 보존) — **Phase 6 검수 큐 라우팅에 사용 가능한 좋은 사실**.
- 문서 단위 분할에 필요한 `document_id` 개념이 `AnnotationPage` 스키마에 없음 — GT 포맷 확장 필요.

## 재사용 가능한 자산

1. `AnnotationDataset` 스키마(픽셀 xywh, 페이지 단위) — GT 포맷으로 재사용 적합. `document_id` 필드만 추가하면 됨.
2. 프론트엔드 어노테이션 좌표계 — 검증 결과 정합. 신뢰하고 사용 가능.
3. `render_pdf_pages` / 데이터 모델 / CLI 뼈대.
4. confidence·raw_label_id 보존 경로.

## 결론 — Phase 1 설계에 대한 요구사항으로 번역

실사 결과가 평가 하네스 설계에 부과하는 제약:

1. **IoU·AP는 전부 신규 구현** (기존 `intersects()` 재사용 금지). 수치 신뢰를 위해 참조 구현(pycocotools) 대조 테스트 필요.
2. **정본 라벨 카탈로그 모듈 신설** — 고정 순서 8라벨로 클래스 id를 한 곳에서만 발급 (C3 해소).
3. **라벨 매핑 계층** — DocStructBench→정본 라벨 매핑 테이블을 명시적 설정으로 (M1 해소, zero-shot 평가 성립 조건).
4. **평가용 예측 생성은 conf=0.001로** — 운영 작동점(0.25)과 평가용 곡선을 구분 (C4 해소).
5. **문서 단위 분할 + 고정 held-out 테스트셋** — `document_id` 추가, 시드 셔플, test 스플릿 (C1·C2 해소). 이것 없이는 이후 모든 실험이 무효이므로 Phase 1 범위에 포함해야 한다.
6. **빈 페이지 처리 정책 명시** — "라벨링 완료" 표시와 "빈 페이지"를 구분 (C5 해소).
7. 운영 비대칭(표 miss ≫ 경계 3px)을 반영한 보조 지표 — 별도 설계안 참조.
