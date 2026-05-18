# Intelligent PDF Parsing Prototype

사용자 어노테이션 기반 PDF 레이아웃 파싱 플랫폼의 프로토타입입니다.

이 코드는 대화에서 언급한 [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO)를 선택적으로 연결할 수 있게 만들었습니다. 모델/가중치가 없을 때도 파이프라인을 검증할 수 있도록 휴리스틱 레이아웃 탐지기를 기본값으로 제공합니다.

## 핵심 기능

- PDF 페이지 렌더링
- 레이아웃 영역 탐지
- 영역 정책 기반 OCR 스킵/수행
- JSON/Markdown 결과 생성
- 사용자 어노테이션을 YOLO 학습 포맷으로 변환
- FastAPI 기반 최소 API
- CLI 기반 로컬 실행

## 설치

기본 API/CLI만 실행:

```bash
pip install -e .
```

PDF 렌더링까지 사용:

```bash
pip install -e ".[pdf]"
```

DocLayout-YOLO 추론까지 사용:

```bash
pip install -e ".[pdf,doclayout]"
```

OCR까지 사용하려면 `pytesseract`와 OS 레벨 Tesseract 설치가 추가로 필요합니다.

```bash
pip install -e ".[pdf,ocr]"
```

## CLI 사용

휴리스틱 탐지기로 파이프라인 검증:

```bash
pdf-layout-parser parse ./sample.pdf --output-dir ./runs/sample
```

DocLayout-YOLO 가중치 사용:

```bash
pdf-layout-parser parse ./sample.pdf \
  --output-dir ./runs/sample \
  --detector doclayout-yolo \
  --model-path ./models/doclayout_yolo_docstructbench_imgsz1024.pt \
  --device cuda:0
```

어노테이션을 YOLO 포맷으로 변환:

```bash
pdf-layout-parser export-yolo ./annotations.json ./dataset
```

10/20페이지 기준 실험 데이터셋과 학습 설정 생성:

```bash
pdf-layout-parser experiment ./annotations.json ./runs/manual-exp \
  --sample-sizes 10,20 \
  --pretrain ./models/doclayout_yolo_docstructbench_imgsz1024.pt \
  --dry-run
```

실제 fine-tuning 실행:

```bash
pdf-layout-parser train ./annotations.json ./runs/manual-train \
  --max-pages 20 \
  --pretrain ./models/doclayout_yolo_docstructbench_imgsz1024.pt \
  --epochs 50 \
  --batch-size 4 \
  --device cuda:0
```

학습된 모델로 다시 파싱:

```bash
pdf-layout-parser parse ./sample.pdf \
  --output-dir ./runs/manual-parse \
  --detector doclayout-yolo \
  --model-path ./runs/manual-train/training/doclayout_finetune/weights/best.pt \
  --device cuda:0
```

## API 실행

```bash
uvicorn pdf_layout_parser.api:app --reload --port 8000
```

웹 UI:

```bash
uvicorn pdf_layout_parser.api:app --reload --port 8000
```

브라우저에서 엽니다.

```text
http://127.0.0.1:8000/
```

웹에서 가능한 기본 기능:

- PDF 업로드
- Heuristic / DocLayout-YOLO detector 선택
- Max page 제한
- OCR 실행 여부 선택
- 페이지 이미지와 탐지 박스 확인
- 영역별 action/text/notes 확인
- 페이지 위 드래그로 어노테이션 박스 추가
- 어노테이션 JSON 다운로드
- `result.json`, `result.md` 확인

엔드포인트:

- `GET /health`
- `POST /documents/parse`
- `POST /annotations/export-yolo`
- `POST /training/fine-tune`

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
          "bbox": {"x": 100, "y": 80, "width": 900, "height": 120}
        }
      ]
    }
  ]
}
```

## 프로토타입 한계

- 실제 성능 검증은 샘플 PDF와 어노테이션 데이터가 필요합니다.
- DocLayout-YOLO 가중치는 별도로 다운로드해야 합니다.
- 현재 API는 MVP 검증용 동기 실행에 가깝습니다. 운영 환경에서는 큐/워커 구조로 분리해야 합니다.

## Fine-tuning 루프

현재 구현된 루프는 다음과 같습니다.

1. PDF 페이지 이미지를 준비합니다.
2. 어노테이션 JSON에 이미지 경로, 페이지 크기, 라벨 박스를 저장합니다.
3. `export-yolo` 또는 `train` 명령으로 YOLO dataset을 생성합니다.
4. `train` 명령이 `doclayout_yolo.YOLOv10(...).train(...)`을 호출합니다.
5. 생성된 `best.pt`를 `parse --detector doclayout-yolo --model-path ...`에 넘깁니다.

`experiment` 명령은 `10,20,50`처럼 페이지 수별 데이터셋과 학습 실행을 나눠 만들기 위한 명령입니다. 기본은 데이터셋과 학습 preview만 만들고, 실제 학습까지 돌리려면 `--train`을 추가합니다.
