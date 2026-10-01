# Decision Record: OCR stack spike (F-DEP-044, F-DEP-045, F-DEP-046)

- **Date:** 2026-10-01
- **Status:** Decided (PDF + barcode swaps implemented here; OCR engine swap deferred)
- **Context:** `pyzbar` (unmaintained ~54 months), `pdf2image` (~32 months) and
  `easyocr` (~24 months) were flagged unmaintained in the MAS modernization
  audit (`F-DEP-044`/`F-DEP-045`/`F-DEP-046`, MODERNIZATION_PLAN task 8.4,
  parent issue COGNIMANEU/pilot03-interface-additive-manufacturing-process-management#214).
  All three also pull in heavyweight or system-level dependencies.

## Method

Candidates were exercised on the real job-card PDFs used by the MAS pilot
(the JCE fixture corpus, `server/data/.../job-cards/original/` in the parent
checkout): `example-02.pdf` (3 pages, GS1 Code-128 operation barcodes, Code-39
job number) and `test-jobcard.pdf` (1 page, Code-128 `J54321`). Host:
Python 3.13.11 on aarch64. Measured output parity, decode coverage, runtime
and packaging footprint.

## Comparison

### PDF → image (`pdf2image`, F-DEP-045)

| Library | Fixture result | Packaging | License | Verdict |
|---|---|---|---|---|
| pdf2image 1.17 (current) | 3 pages, 1653×2339 @200 dpi, 0.26 s; 1 page, 0.11 s | Requires system **poppler** binaries | MIT | Replace |
| **pypdfium2 5.13** | Identical page count and geometry (1653×2339 @200 dpi), faster (0.20 s / 0.04 s) | Self-contained wheel | Apache-2.0/BSD-3 | **Adopt** |
| pymupdf 1.28 | Identical output, fast (0.23 s / 0.03 s) | Self-contained wheel | **AGPL-3.0**/commercial | Rejected (license) |

### Barcode decode (`pyzbar`, F-DEP-044)

| Library | example-02 | test-jobcard | Packaging | Verdict |
|---|---|---|---|---|
| pyzbar 0.1.9 (current) | 26 hits | 1 hit | Requires system **libzbar**; the current venv needed a machine-local `ctypes` shim to load it | Replace |
| **zxing-cpp 3.1.1** | **33 hits — strict superset**, incl. job number `4440801` and all GS1 operation barcodes | 1 hit (same) | Self-contained wheel | **Adopt** |
| opencv `cv2.barcode` | 0 hits | 0 hits | Already a dependency | Rejected (no Code-39/128 detection on fixtures) |

pyzbar emits raw control bytes (`\x1e`/`\x1f`) in `data`; zxing-cpp `.bytes`
returns equivalent raw content (its HRI text renders them `<RS>`/`<US>`).
`clean_barcode_value` strips these, so extracted values are unchanged.

### OCR engine (`easyocr`, F-DEP-046)

| Library | example-02 | test-jobcard | Weight | Verdict |
|---|---|---|---|---|
| easyocr 1.7.2 (current) | 329 lines / 49.8 s | 14 lines / 5.8 s | torch+torchvision (~GB) | **Keep for now** |
| rapidocr-onnxruntime 1.2.3 | 212 lines / 5.0 s — same fields found, different line segmentation (e.g. `2022-Inspection` merged vs split) | 11 lines / 0.84 s | onnxruntime (~100 MB) | Designated successor, deferred |

RapidOCR is ~10× faster and removes the entire torch stack, but its text-line
segmentation differs from EasyOCR's `readtext` output that the extraction
regexes were tuned against. Swapping the OCR engine is a **quality risk on
real cards**, not a drop-in — deferred until the JCE fixture-test task
(parent #225, "9.10") lands a labelled corpus to validate parity.

## Decision

1. **`pdf2image` → `pypdfium2>=4.30`** — implemented here behind a
   `convert_from_path()` shim (200 dpi default, identical PIL output). Removes
   the system poppler dependency.
2. **`pyzbar` → `zxing-cpp>=2.2`** — implemented here behind a `decode()` shim
   returning pyzbar-shaped results (`data` bytes, `type` normalised to
   `CODE128`-style names, `rect` x/y/w/h, `quality`). Removes the system
   libzbar dependency and its host-specific load shim.
3. **`easyocr` kept** (`rapidocr-onnxruntime` validated as the successor
   candidate) — swap deferred pending extraction-quality validation on a
   labelled fixture corpus (parent #225).

## Consequences

- `requirements.txt`/`pyproject.toml` now declare `pypdfium2` and `zxing-cpp`
  instead of `pdf2image`/`pyzbar`; `requirements.lock` regenerated via
  `uv pip compile` (hash-pinned).
- `install.sh` no longer installs poppler or needs sudo; CI no longer
  installs `libzbar0`.
- Test mocks target the module-level `convert_from_path`/`decode` names,
  which are preserved — the suite stays at 37 passed (≥ the 27 baseline).
- Follow-up: evaluate `rapidocr(-onnxruntime)` vs EasyOCR extraction parity
  once labelled fixture tests exist; then drop torch/torchvision (~GB
  install reduction).
