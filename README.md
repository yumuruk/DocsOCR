# DocsOCR PDF Layout Parser

## Quick Start: 처음 실행

아래 명령은 GitHub repo 이름과 동일한 로컬 폴더 `DocsOCR` 기준입니다.

```powershell
cd "C:\Users\taeil\Projects\DocsOCR"
```

### 1. 처음 한 번만 설치

```powershell
uv sync --extra pdf --extra doclayout
cd frontend
npm install
cd ..
```


### DocStructBench 기본 모델 배치

DocLayout-YOLO의 DocStructBench fine-tuned 모델은 레포에 커밋하지 않고 로컬 `models/` 폴더에 둡니다.

```powershell
mkdir models
copy "C:\Users\taeil\Downloads\doclayout_yolo_docstructbench_imgsz1024.pt" "models\doclayout_yolo_docstructbench_imgsz1024.pt"
```

웹 UI의 기본 detector는 `doclayout-yolo`이고, 기본 model path는 아래 파일입니다.

```text
models/doclayout_yolo_docstructbench_imgsz1024.pt
```
### 2. 터미널 1: 백엔드 실행

반드시 프로젝트 루트에서 실행해야 합니다. `frontend` 폴더 안에서 실행하면 `No module named 'backend'` 오류가 납니다.

```powershell
uv run uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

### 3. 터미널 2: 프론트엔드 실행

```powershell
cd frontend
npm run dev
```

### 4. 브라우저 접속

```text
http://127.0.0.1:5173/
```

사용자 어노테이션 기반 PDF 레이아웃 파싱 프로토타입입니다.

이 프로젝트는 [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO)를 선택적으로 연결할 수 있게 구성되어 있습니다. 모델 가중치가 없어도 기본 `heuristic` detector로 PDF 업로드, 페이지 렌더링, 영역 박스 시각화, 어노테이션 export 흐름을 확인할 수 있습니다.

## 주요 기능

- PDF 업로드 및 페이지 이미지 렌더링
- 레이아웃 영역 탐지
- 영역 정책 기반 OCR 실행/스킵
- JSON/Markdown 결과 생성
- 페이지 위 드래그 어노테이션
- 어노테이션 JSON export
- YOLO 학습 데이터셋 export
- DocLayout-YOLO fine-tuning 래퍼
- FastAPI 백엔드
- React + Vite + Tailwind 프론트엔드

## 프로젝트 구조

```text
.
├── frontend/                   # React + Vite + Tailwind 웹 UI
├── backend/                    # FastAPI 백엔드 서버
├── src/pdf_layout_parser/      # OCR/레이아웃 핵심 Python 모듈
├── docs/                       # 기획/요구사항 문서
├── tests/                      # Python 테스트
├── pyproject.toml              # Python 패키지 설정
├── requirements.txt            # Python 기본 의존성
└── README.md
```

핵심 위치는 아래처럼 보면 됩니다.

| 영역 | 위치 | 역할 |
| --- | --- | --- |
| Frontend | `frontend/` | React UI, PDF 업로드 화면, 박스 시각화, 어노테이션 export |
| Backend | `backend/main.py` | FastAPI API 서버, 파일 업로드, 파싱 요청, 결과 파일 서빙 |
| Python core module | `src/pdf_layout_parser/` | PDF 렌더링, 레이아웃 탐지, OCR, 후처리, YOLO export/fine-tuning |

Python core module 안의 주요 파일:

- `pipeline.py`: PDF 파싱 전체 파이프라인
- `pdf.py`: PDF 페이지 렌더링 및 native text 추출
- `detectors.py`: heuristic detector 및 DocLayout-YOLO detector adapter
- `ocr.py`: OCR 실행 adapter
- `dataset.py`: 어노테이션 JSON -> YOLO dataset 변환
- `training.py`: DocLayout-YOLO fine-tuning wrapper
- `models.py`: 공통 데이터 모델
- `labels.py`: 정본 라벨 카탈로그, 클래스 id 발급, DocStructBench 라벨 매핑
- `splits.py`: 문서 단위 시드 분할, 고정 held-out 테스트셋, 중첩 표집
- `evaluation/`: 평가 하네스 (IoU/AP, 운영 지표, 리포트)
- `experiment_log.py`: append-only 실험 원장 기록
- `core/__init__.py`: 핵심 파이프라인 import facade

라벨 클래스 id는 `labels.CANONICAL_LABELS`에서만 발급됩니다. 어노테이션 JSON의 `labels` 배열
순서에서 id를 유도하면 파일마다 id가 달라져 데이터셋 간 모델 호환이 깨집니다.

## 필수 의존성

### 1. Python

- Python 3.10 이상 권장
- 현재 개발 환경에서는 `uv`로 Python 3.12 가상환경을 사용했습니다.

`uv`가 있으면 아래 명령을 그대로 사용할 수 있습니다.

```bash
uv run python --version
```

`uv` 없이 일반 Python을 사용할 수도 있습니다.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[pdf]"
```

### 2. Node.js / npm

React/Vite 프론트엔드를 실행하려면 Node.js와 npm이 필요합니다.

- Node.js 20 이상 권장
- 현재 개발 환경에서는 Node.js 24 LTS를 사용했습니다.

확인:

```bash
node --version
npm --version
```

### 3. 선택 의존성

DocLayout-YOLO 추론/학습을 사용하려면:

```bash
pip install -e ".[pdf,doclayout]"
```

OCR까지 사용하려면:

```bash
pip install -e ".[pdf,ocr]"
```

`pytesseract`를 쓰는 경우 OS에 Tesseract OCR 설치가 추가로 필요합니다.

## 처음 실행하기

### 1. Python 백엔드 의존성 설치

권장:

```bash
uv sync --extra pdf --extra doclayout
```

또는 pip:

```bash
pip install -e ".[pdf]"
```

### 2. React 프론트엔드 의존성 설치

```bash
cd frontend
npm install
cd ..
```

## 개발 모드 실행

터미널을 2개 열어서 백엔드와 프론트엔드를 각각 실행합니다.

### 터미널 1: FastAPI 백엔드

```bash
uv run uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

또는 pip 환경:

```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

백엔드 확인:

```text
http://127.0.0.1:8000/health
```

### 터미널 2: React/Vite 프론트엔드

```bash
cd frontend
npm run dev
```

브라우저에서 접속:

```text
http://127.0.0.1:5173/
```

Vite dev server는 아래 요청을 FastAPI 백엔드로 프록시합니다.

- `/documents`
- `/annotations`
- `/training`
- `/runs`
- `/health`

## 빌드 후 FastAPI에서 서빙하기

React 앱을 정적 파일로 빌드합니다.

```bash
cd frontend
npm run build
cd ..
```

FastAPI를 실행합니다.

```bash
uv run uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

브라우저에서 접속:

```text
http://127.0.0.1:8000/
```

`frontend/dist`가 존재하면 FastAPI 루트(`/`)에서 React 빌드 결과를 서빙합니다. 빌드 결과가 없으면 기존 정적 HTML 프로토타입을 fallback으로 서빙합니다.

## 웹 UI 사용법

1. `http://127.0.0.1:5173/` 접속
2. PDF 파일 선택
3. Detector 선택
   - `heuristic`: 모델 없이 기본 박스 생성
   - `doclayout-yolo`: DocLayout-YOLO 가중치 사용
4. 필요하면 `Max pages` 지정
5. OCR 실행 여부 선택
6. `Run parse` 클릭
7. 페이지 이미지와 탐지 박스 확인
8. label 선택 후 페이지 위에서 드래그해 어노테이션 박스 추가
9. `Export`로 `annotations.json` 다운로드
10. `result.json`, `result.md` 결과 확인

## CLI 사용

기본 heuristic detector로 파싱:

```bash
uv run pdf-layout-parser parse ./sample.pdf --output-dir ./runs/sample --no-ocr
```

DocLayout-YOLO 가중치로 파싱:

```bash
uv run pdf-layout-parser parse ./sample.pdf ^
  --output-dir ./runs/sample ^
  --detector doclayout-yolo ^
  --model-path ./models/doclayout_yolo_docstructbench_imgsz1024.pt ^
  --device cuda:0
```

어노테이션 JSON을 YOLO 데이터셋으로 변환:

```bash
uv run pdf-layout-parser export-yolo ./annotations.json ./dataset
```

10/20페이지 기준 실험 데이터셋과 학습 preview 생성:

```bash
uv run pdf-layout-parser experiment ./annotations.json ./runs/manual-exp ^
  --sample-sizes 10,20 ^
  --pretrain ./models/doclayout_yolo_docstructbench_imgsz1024.pt ^
  --dry-run
```

실제 fine-tuning 실행:

```bash
uv run pdf-layout-parser train ./annotations.json ./runs/manual-train ^
  --max-pages 20 ^
  --pretrain ./models/doclayout_yolo_docstructbench_imgsz1024.pt ^
  --epochs 50 ^
  --batch-size 4 ^
  --device cuda:0
```

학습된 모델로 다시 파싱:

```bash
uv run pdf-layout-parser parse ./sample.pdf ^
  --output-dir ./runs/manual-parse ^
  --detector doclayout-yolo ^
  --model-path ./runs/manual-train/training/doclayout_finetune/weights/best.pt ^
  --device cuda:0
```

## 평가 하네스

모델 성능을 측정하고 실험을 기록하는 부분입니다. 측정 규칙 정의는
[docs/EVALUATION.md](docs/EVALUATION.md), 실측 원장은 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md),
판단 경위는 [docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md)에 있습니다.

held-out 테스트셋에서 채점:

```bash
uv run pdf-layout-parser evaluate ./annotations.json ./runs/eval ^
  --run-name ft-n20-s0 ^
  --detector doclayout-yolo ^
  --model-path ./runs/manual-train/training/doclayout_finetune/weights/best.pt ^
  --label-mapping identity ^
  --split test --split-seed 0 ^
  --tag init=docstructbench --tag n=20 --tag labels=manual
```

사전학습 가중치의 zero-shot 베이스라인은 라벨 매핑이 **필수**입니다. DocStructBench 클래스명
(`plain text`, `abandon`, ...)은 레포의 정본 8라벨과 다르므로 매핑 없이는 per-class AP가 성립하지 않습니다.

```bash
uv run pdf-layout-parser evaluate ./annotations.json ./runs/eval ^
  --run-name zeroshot ^
  --detector doclayout-yolo ^
  --model-path ./models/doclayout_yolo_docstructbench_imgsz1024.pt ^
  --label-mapping docstructbench ^
  --tag init=zeroshot
```

여러 run을 비교표로:

```bash
uv run pdf-layout-parser compare ./runs/eval --output docs/comparison.md
```

리포트는 `<run-name>.md` / `<run-name>.json`으로 저장되고, 예측은
`<run-name>_predictions.json`에 남습니다. 지표 코드를 수정한 뒤 모델 없이 과거 run을 재채점하려면
`--predictions <path>`를 씁니다.

`evaluate`는 기본으로 `docs/EXPERIMENTS.md`에 행을 추가합니다(`--no-ledger`로 끕니다).

측정되는 것:

- **운영 지표** — 절감률(전체 페이지 OCR 대비 면적 절감), 유실률(GT 내용 중 버려진 비율),
  오염률(추출 영역 중 내용이 아닌 비율). `full_page_ocr`·`oracle` 기준선이 같은 표에 들어갑니다.
- **검출 지표** — per-class AP@0.5 / AP@0.5:0.95, mAP, 혼동행렬. COCO 관례이며 `pycocotools`와
  대조 검증돼 있습니다.

주의: 평가용 예측은 conf **0.001**로 생성하고(PR 곡선 recall 꼬리 보존), 운영 지표와 혼동행렬만
작동점 **0.25**를 씁니다. 이 둘을 같은 값으로 두면 AP가 체계적으로 과소평가됩니다.

## 웹에서 fine-tuning (quick loop)

어노테이션 → 학습 → 전/후 비교를 웹에서 한 번에 돌릴 수 있습니다.

1. 파싱 후 박스를 그리거나 `Predictions`로 모델 출력을 불러와 수정
2. 페이지마다 **Labeled** 체크 (최소 3페이지 — train/val/test로 분할됨)
3. `Fine-tune & compare` 클릭 → 백그라운드 잡이 학습 후 **같은 held-out 페이지에서**
   baseline(사전학습 zero-shot)과 tuned 모델을 채점
4. 전/후 테이블(mAP@.5 / save / loss / noise) 확인 → `Use tuned model`로 경로 주입 →
   `Run parse`로 tuned 박스 확인

주의: 웹 경로는 문서 1개 기준 page-level 분할이라 같은 문서의 안 본 페이지에 대한
숫자입니다. 문서 단위로 분할하는 CLI 실험(`experiment`/`evaluate`)과 수치를 비교하지
마세요. 학습은 서버와 분리된 서브프로세스에서 돌고, 결과는
`runs/web-training/<job>/comparison.json`에 남습니다.

## API 엔드포인트

- `GET /health`
- `POST /documents/parse`
- `POST /annotations/export-yolo`
- `POST /training/fine-tune` — JSON body(어노테이션 payload + epochs/device/seed),
  즉시 `job_id` 반환
- `GET /training/jobs/{job_id}` — 잡 상태/단계/결과 폴링

FastAPI Swagger 문서:

```text
http://127.0.0.1:8000/docs
```

## 어노테이션 JSON 예시

```json
{
  "labels": ["title", "body", "table", "figure", "header", "footer", "ignore", "target_data"],
  "pages": [
    {
      "image_path": "runs/sample/pages/page_0001.png",
      "width": 1240,
      "height": 1754,
      "annotations": [
        {
          "label": "title",
          "bbox": {
            "x": 100,
            "y": 80,
            "width": 900,
            "height": 120
          }
        }
      ]
    }
  ]
}
```

## 테스트 및 검증

Python 테스트:

```bash
uv run python -m unittest discover tests
```

프론트엔드 빌드:

```bash
cd frontend
npm run build
```

현재 검증된 명령:

```bash
uv run python -m unittest discover tests
cd frontend
npm install
npm run build
```

## 참고 사항

- `runs/`, `.venv/`, `.uv-cache/`, `.uv-python/`, `frontend/node_modules/`, `frontend/dist/`는 git에 포함하지 않습니다.
- 실제 DocLayout-YOLO 사용을 위해서는 별도 가중치 파일이 필요합니다.
- 현재 API는 MVP 검증용 동기 실행 구조에 가깝습니다. 운영 환경에서는 job queue/worker 구조로 분리하는 것이 좋습니다.
- OCR 품질은 PDF 유형, 이미지 해상도, OCR 엔진 설치 여부에 따라 달라집니다.

