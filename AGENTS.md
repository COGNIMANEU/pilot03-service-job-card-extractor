# Job Card Extractor — delegatable specialists

## Extraction implementation specialist
- Scope: job details, operations, OCR/barcode processing, PDF pipeline, and CLI behavior in `job_card_extractor.py`.
- Boundary: preserve the imported API and result shape described in `docs/api-reference.md`; report required test or documentation changes to the respective specialist rather than editing their files.

## Regression-test specialist
- Scope: focused coverage in `tests/test_job_extraction.py`, `tests/test_ocr.py`, `tests/test_barcode_extraction.py`, `tests/test_processing.py`, and `tests/test_version.py`.
- Boundary: mock PDF/OCR dependencies in unit tests; report implementation defects without editing `job_card_extractor.py`.

## API and usage documentation specialist
- Scope: consumer-facing examples and result-contract updates in `README.md`, `docs/api-reference.md`, and `docs/user-guide.md`.
- Boundary: check claims against `job_card_extractor.py`; leave environment and test commands in `CLAUDE.md`, and do not edit implementation or tests.
