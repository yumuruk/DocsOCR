# DocsOCR PDF Layout Parser

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
- `core/__init__.py`: 핵심 파이프라인 import facade

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
uv sync --extra pdf
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

## API 엔드포인트

- `GET /health`
- `POST /documents/parse`
- `POST /annotations/export-yolo`
- `POST /training/fine-tune`

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
