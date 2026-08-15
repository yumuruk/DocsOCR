# Research log

날짜별 서술 기록: 무엇을 물었고, 무엇을 봤고, 그 결과로 무엇을 바꿨는지.
숫자는 [EXPERIMENTS.md](EXPERIMENTS.md) 원장에, 측정 규칙은 [EVALUATION.md](EVALUATION.md)에 있다.
최신 항목이 아래쪽.

## 2026-08-08 — Phase 0 코드 실사

전체 내용은 [../FINDINGS.md](../FINDINGS.md). 영역별 감사 6개를 병렬 수행한 뒤 critical·major 주장
전건을 재검증했다. Phase 1 설계에 영향을 준 결론만 요약:

- **평가 코드 전무.** mAP·IoU·AP를 계산하는 코드가 한 줄도 없다. `models.py:44` `intersects()`는
  IoU가 아니라 `intersection/min(area)`이고, `training.py:120` `extract_metrics()`는 ultralytics
  `results_dict` pass-through다.
- **실험 구조가 비교 불가.** 샘플링이 prefix 슬라이스이고 N마다 80/20 분할을 다시 하므로 N별 val이
  다르다. 공통 held-out 테스트셋이 없다.
- **조용한 오염원 4개.** 클래스 id가 파일 종속, conf 0.25가 predict 시점 적용, 빈 페이지가 배경
  negative로 export, DocStructBench↔레포 라벨 매핑 부재.
- **HITL 실험 불가.** UI가 박스 신규 그리기 + 전체 삭제만 지원한다.

정합 확인된 자산: 프론트엔드 export 좌표계와 `to_yolo_bbox()`의 기대가 정확히 일치한다.

## 2026-08-15 — 방법론 정리: 두 과제를 분리

논의 중에 드러난 것 하나가 지표 설계를 바꿨다.

레이아웃 모델이 완전히 동일한 구조가 아니어도 스킵은 상당히 일반화된다는 관찰이 맞다.
스킵은 위치·텍스트 밀도·종횡비 같은 저수준 단서로 결정되는 **거친 과제**라서 레이아웃이 달라도
전이되고, 실제로 DocStructBench가 `abandon`/`figure`를 zero-shot으로 그럭저럭 잡는다.
반면 `target_data`는 "이 문서군에서 이 필드가 여기 있다"는 고수준 관습에 의존해 일반화가 약하다.

**결론: 한 프로젝트 안에 난이도가 다른 두 과제가 섞여 있고 단일 mAP는 이를 평균으로 뭉갠다.**
그리고 스킵 과제에는 mAP가 애초에 맞는 지표가 아니다 — 박스 경계 3px이 아니라 "버릴 걸 버렸나,
살릴 걸 죽였나"가 전부다. 그래서 면적 기반 운영 지표(절감/유실/오염)를 별도로 정의했다.
정의와 목적함수는 EVALUATION.md §2.

부수적으로 정리된 것: 유실과 절감은 교환 대상이 아니다. 유실률에 하드 제약을 걸고 그 아래에서
절감률을 최대화하는 것이 목적함수다. mAP가 이겨도 유실률이 크면 탈락이다.

**정직한 유보 하나.** "반복 헤더 제거"만 놓고 보면 페이지 간 반복 검출(cross-page repetition)로
학습 없이도 잡힌다. fine-tuning의 진짜 정당화는 위치·반복으로 규정되지 않는 `table`/`target_data`여야
한다. 이건 아직 측정된 바 없고, 순차 축소 1단계에서 답이 나와야 하는 질문이다.

## 2026-08-15 — Phase 1: 평가 하네스 구현

FINDINGS 결론부의 요구사항 7개를 코드로 옮겼다. 신규: `labels.py`, `splits.py`,
`evaluation/` 패키지, `experiment_log.py`. 수정: `dataset.py`, `models.py`, `cli.py`.

| FINDINGS 요구사항 | 구현 |
|---|---|
| IoU·AP 신규 구현 + 참조 대조 | `evaluation/detection.py`, pycocotools 대조 통과 |
| 정본 라벨 카탈로그 (C3) | `labels.CANONICAL_LABELS`, `data.yaml`이 항상 nc=8 |
| 라벨 매핑 계층 (M1) | `labels.DOCSTRUCTBENCH_TO_CANONICAL`, 리포트에 매핑명 기록 |
| 평가용 conf 0.001 (C4) | `EVAL_CONFIDENCE` / `OPERATING_CONFIDENCE` 구조적 분리 |
| 문서 단위 분할 + 고정 test (C1·C2) | `splits.make_splits`, `document_id`, 중첩 표집 |
| 빈 페이지 정책 (C5) | `LabelStatus`, complete만 채택 |
| 운영 비대칭 보조 지표 | `evaluation/operational.py` 절감/유실/오염 |

**설계 판단 몇 가지와 근거:**

- **순수 파이썬.** 기존 테스트가 선택 의존성 없이 통과하는 성질을 유지해야 해서 numpy/torch에
  의존하지 않았다. 면적 계산은 픽셀 마스크 대신 좌표 압축으로 정확히 구한다. 1240×1754 페이지가
  마스크로는 2.2M 픽셀이지만 압축 격자는 수천 셀이다.
- **`AnnotationSource` 태깅.** self-training과 HITL을 같은 지표로 비교하려면 라벨 출처가 데이터에
  남아야 한다. export 후에는 구분이 불가능해진다.
- **레거시 prefix 분할을 삭제하지 않고 경고 + `strict=True`로 차단.** 데이터셋을 눈으로 확인하는
  용도는 남겨두고, 숫자가 중요한 경로에서만 막았다. 단일 페이지가 train과 val 양쪽에 들어가던
  `return pages, pages`는 무조건 예외로 바꿨다.
- **`nc`를 항상 8로 고정.** 데이터셋에 등장하는 클래스에 맞춰 크기를 정하는 것이 id 드리프트의
  원인이었다.
- **oracle 절감률 == full-page 오염률** 항등식을 테스트로 고정했다(둘 다 `1 - gt_keep/page`).
  면적 계산이 틀리면 여기서 깨진다.

검증: 74 tests OK. pycocotools 대조 3건(AP@0.5, 0.50:0.95 스윕, 다중 클래스) 통과.
합성 fixture(5문서×3페이지)로 CLI E2E 확인.

**E2E에서 M4가 재확인됐다.** heuristic 베이스라인은 `table`을 한 번도 예측하지 않아 AP 0이고,
header/footer는 6번씩 예측했는데도 AP 0이다 — 고정 밴드가 높이의 8%(70px)인데 GT는 40px라
IoU가 0.5를 못 넘긴다. 혼동행렬이 이를 `missed` + `__background__` 예측으로 정확히 보여준다.
하한선으로서는 제 역할을 한다.

**남은 차단 요인 (실측 시작 전에 풀어야 함):**

1. **라벨링 예산 ~90페이지.** 고정 테스트셋 30~50 + 학습 풀 50+, 문서 최소 3개. `n=20`만
   라벨링하면 실험이 성립하지 않는다.
2. **UI 보강이 Phase 1의 선행 조건일 수 있다.** 개별 박스 삭제·이동·리사이즈·리라벨이 없는 상태로
   90페이지를 그리는 것은 실수 1번에 전체 삭제라는 리스크를 안는다(M2).
3. **`training.py`에 `seed`·`freeze` 미노출.** 작은 n에서는 시드 분산이 측정하려는 효과보다 크므로
   3시드 반복이 필수인데 현재 시드를 지정할 수 없다. `freeze` 없이는 백본 동결 축도 못 돈다.
4. **`mosaic` 기본값 1.0.** 10페이지 데이터셋에 과할 가능성이 있어 실험 축 후보로 올렸다.

## 2026-08-15 — 적대적 자체 검토: 하네스에서 결함 5건 발견·수정

Phase 1 코드를 실측 시작 전에 적대적으로 재검토했다. 방법: 각 모듈에 실패 시나리오를
실제로 실행해 공격 (극단 코퍼스 분할, 무작위 200케이스 기하 대조, 동점 score 결정성,
GT 밖 페이지 예측 주입, 파라미터 무시 여부). 기하 코어(브루트포스 픽셀 카운트 대비
오차 0)와 AP(pycocotools 대조 유지)는 통과했고, 5건이 걸렸다.

**결함과 수정:**

1. **분할 배정이 크기 편차 코퍼스에서 붕괴 (critical).** greedy가 test→val을 먼저
   채우고 train은 잔여만 받아, "매뉴얼 60p + 짧은 문서 5개" 같은 전형적 코퍼스에서
   20시드 중 10회 SplitError, 성공해도 train 3~12p (n=50 셀 불가능). → 크기 내림차순
   배치 + 상대 미달률 최소 split 배정 + 빈 split 수리 + 편차 경고로 교체. 5개 코퍼스
   전부 20/20 성공, 매뉴얼 코퍼스 train 60p. 부수 효과로 test 배정이 크기 상이 문서에서
   시드 불변이 됐는데, 이는 "동결 test셋" 원칙과 오히려 정합해 채택.
2. **`train_pool`이 val 포함 (critical).** 표집된 학습 페이지가 early stopping이 모델을
   고르는 val에 들어가는 누수. → train만 반환하도록 수정, 회귀 테스트 추가.
3. **GT 밖 페이지 예측이 AP를 조용히 깎음.** detection 매처가 GT 없는 페이지의 예측을
   전부 FP로 계산 (실측: AP 1.0→0.5). CLI 경로는 안전하나 `--predictions` 캐시 재채점
   경로가 위험했고, note 문구("운영 지표에서 무시")가 detection 동작과 상충. →
   `evaluate()`가 채점 전 GT 페이지 밖 예측을 필터링하고 제외 건수를 note에 기록.
4. **`evaluate_detections(score_threshold=)`가 필터가 아닌데 필터처럼 읽힘.** 동작은
   의도대로(AP는 전체 예측 사용)였으나 이름이 오독 유발. → `recorded_score_floor`로
   개명, "provenance이지 필터가 아니다" docstring 명시.
5. **`build_data_yaml(labels=)` 죽은 파라미터.** 받아서 조용히 무시. → 파라미터 제거.

95 tests OK. 교훈: 1·2번 모두 "정상 케이스 테스트는 전부 통과하는" 결함이었다 —
균등 크기 문서로만 테스트했기 때문. 회귀 테스트는 이제 불균형 코퍼스(UnevenCorpusTests)를
포함한다. 실측 데이터를 라벨링하기 전에 잡았으므로 무효화된 run은 없다.

## 2026-08-15 — training.py 보강: 재현성 표면 완성 (차단 요인 #3 해소)

M3의 공통 형태는 "run은 돌았지만 기록이 무엇이 돌았는지 말하지 않는다"였다. 수정:

- **`seed`·`freeze` 노출** — `TrainingConfig`와 CLI(`--seed`, `--freeze N`). 작은 n 셀의
  3시드 반복과 백본 동결 실험 축이 이제 가능하다. `freeze=None`은 ultralytics에 아예
  전달하지 않는다 (부재가 "전체 학습"의 표현).
- **기록 == 실행 보장** — `training_kwargs()`가 단일 진실원: 실제 `YOLOv10.train(**kwargs)`
  호출, `TrainingResult.train_kwargs`, preview 문자열이 전부 같은 dict에서 나온다. 기존
  preview는 손으로 조립돼 seed·freeze·lr 등을 누락했다.
- **디스크 저장** — 모든 run(dry 포함)이 `<run_name>_result.json`을 남기고, `experiment`는
  `experiments.json` manifest를 쓴다. 기존에는 stdout이 유일한 기록이었다.
- **학습 시간 기록**, **from-scratch 경고** — `pretrain` 미지정 시 랜덤 초기화라는 사실을
  결과에 명시 (기존엔 run_name이 "finetune"인 채 조용히 from-scratch).
- **experiment 루프를 splits로 재배선** — 기존 루프는 여전히 legacy prefix 분할을 써서
  C2가 실험 명령 경로에는 남아 있었다. 이제 분할을 한 번 만들고 모든 N 셀이 동일한
  val/test를 공유하며 train만 중첩 표집한다 (`with_train_sample`). strict export라서
  안전하지 않은 분할이면 실험이 시작 전에 거부된다. run_name에 학습 시드 포함
  (`doclayout_pages_20_seed1`) — exist_ok=True에서도 시드 반복이 서로 덮어쓰지 않는다.
  `--train-ratio` 플래그는 experiment에서 제거 — N별 재분할이 바로 C2였다.

E2E 확인: `experiment --sample-sizes 3,6 --seed 1 --freeze 10` dry run에서 N=3/N=6의
val.txt·test.txt 동일, result json에 seed/freeze 기록. 105 tests OK.

**실측 차단 요인 현황**: #3(seed/freeze) 해소. 남은 것은 #1(라벨링 ~90페이지),
#2(어노테이션 UI — removeAnnotation은 들어갔고 이동·리사이즈·리라벨·재로드 필요),
#4(mosaic 기본값 — 이제 실험 축으로 돌릴 수 있음).
