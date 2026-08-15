# Experiment ledger

Append-only record of every scored run. Rows are added by
`pdf_layout_parser.experiment_log.append_run` (the `evaluate` CLI does this by
default). **Do not edit or delete rows.** If a run turns out to have been
misconfigured, add a new row and say so in `RESEARCH_LOG.md` -- the old row is
still what happened.

Column meanings are defined in [EVALUATION.md](EVALUATION.md). Short version:
`reduction` = OCR area saved vs full-page, `loss` = ground truth content
discarded (hard constraint), `contam` = share of extracted area that is not
content.

`data` is a content hash of **the pages that were scored**, not of the whole
corpus. Two rows sharing it were scored on byte-identical ground truth and can be
compared directly. Two rows differing in it were not -- usually because the split
seed differs, which makes them replicates rather than comparable cells. Check this
column before reading any difference between rows as a model difference.

**현재 실측 run 0건.** 하네스는 합성 fixture로 E2E 검증했으나 라벨링된 코퍼스가 아직 없다.
실측 시작에 필요한 라벨링 예산(~90페이지)과 차단 요인은
[RESEARCH_LOG.md](RESEARCH_LOG.md) 2026-08-15 항목을 참조.

| # | Date | Run | Cell | Split | Pages | mAP@.5 | mAP@.5:.95 | Reduction | Loss | Contam | data | seed | Report |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
<!-- ledger:rows -->
