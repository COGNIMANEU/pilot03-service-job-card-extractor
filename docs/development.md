# Development

Local setup, testing, and contribution guide for Job Card Extractor.

## Project Structure

```
pilot03-service-job-card-extractor/
├── job_card_extractor.py       # Main application (single-file CLI tool)
├── pyproject.toml              # Python version (3.13), dependencies, ruff config
├── requirements.txt            # Loose dependency floors (used by the MAS installer)
├── requirements.lock           # Hash-pinned runtime dependencies (Python 3.13)
├── requirements-dev.in         # Test/lint tool inputs (pytest, ruff)
├── requirements-dev.lock       # Hash-pinned test/lint tools
├── README.md                   # Project entry point
├── CLAUDE.md                   # AI agent context
├── tests/                      # Unit tests (pytest)
│   ├── test_version.py         # Version function tests
│   ├── test_job_extraction.py  # Job details and operation extraction
│   ├── test_ocr.py             # OCR and image preprocessing
│   ├── test_barcode_extraction.py  # Barcode detection and cleaning
│   └── test_processing.py      # Main pipeline and CLI tests
├── samples/                    # Example PDFs for manual testing
│   ├── example-01.pdf
│   └── example-02.pdf
├── docs/                       # Documentation
│   ├── architecture.md
│   ├── user-guide.md
│   ├── api-reference.md
│   ├── development.md          # (this file)
│   └── troubleshooting.md
└── output/                     # Generated output (git-ignored)
```

## Local Setup

### Prerequisites

- Python 3.13

No system dependencies: PDF rendering uses `pypdfium2` and barcode reading
uses `zxing-cpp`, both shipped as self-contained wheels (see
`docs/decisions/ocr-stack-spike.md`).

### Installation

```bash
git clone https://github.com/COGNIMANEU/pilot03-service-job-card-extractor
cd pilot03-service-job-card-extractor

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies (hash-verified) and the test/lint tools
pip install --require-hashes -r requirements.lock
pip install --no-deps --require-hashes -r requirements-dev.lock
```

First run will download EasyOCR language models (~100MB+). Ensure internet connectivity.

### Verify Installation

```bash
# Check version
python job_card_extractor.py --version

# Run on sample file
python job_card_extractor.py samples/example-01.pdf -o output
```

## Dependency Locks

`pyproject.toml` declares `requires-python = ">=3.13,<3.14"` and the direct
dependencies (kept identical to `requirements.txt`; a test checks this).
`requirements.lock` pins the full universal (Linux, macOS, Windows) resolution
with sha256 hashes for Python 3.13, and `requirements-dev.lock` pins pytest and
ruff separately so the runtime environment never carries test tools. Both are
generated, not hand-edited, with [uv](https://docs.astral.sh/uv/):

```bash
uv pip compile pyproject.toml --universal --python-version 3.13 \
    --generate-hashes -o requirements.lock
uv pip compile requirements-dev.in --universal --python-version 3.13 \
    --generate-hashes -c requirements.lock -o requirements-dev.lock
```

Install with `--require-hashes` (the dev lock also with `--no-deps`, as it is
fully resolved). The parent MAS repository installs `requirements.lock` and then
its own hash-pinned pytest (`requirements/python-test-tools.lock`); the dev lock
uses the same pytest version so the two never disagree. Add `--upgrade` to the
first command only when deliberately refreshing versions.

## Running Tests

```bash
# Activate virtual environment
source .venv/bin/activate

# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_job_extraction.py

# Run with verbose output
pytest tests/ -v
```

Tests use `unittest.mock` to avoid requiring actual PDF files or OCR processing during testing.

### Test Coverage

| File | Scope |
|------|-------|
| `test_version.py` | Version retrieval and display |
| `test_job_extraction.py` | Job details, operations, combined extraction |
| `test_ocr.py` | Image preprocessing, OCR, debug images |
| `test_barcode_extraction.py` | Barcode cleaning, detection, line detection |
| `test_processing.py` | Main pipeline and CLI argument handling |
| `test_failure_semantics.py` | Loud-failure exit codes and propagation |
| `test_cleanup_and_matching.py` | Barcode matching correctness, 80-line limit, CLI flags, cache safety, debug naming |

## Code Organization

The project is a single-file application (`job_card_extractor.py`) organized into sections
(names rather than line numbers — the layout shifts as the module evolves):

| Section | Description |
|---------|-------------|
| Constants | Named thresholds and pattern tables (`MIN_*`, `*_PATTERNS`) |
| `ExtractionLogger` | Logging system for tracking extraction process |
| Version functions | `get_version()`, `display_version()` |
| Barcode & OCR | Detection, preprocessing, thread-safe OCR cache |
| PDF processing | Page processing and parallel/sequential PDF extraction |
| Job & operations | Extraction logic with pattern matching |
| Main processing | Entry point: `process_pdf_document()` |
| CLI interface | `_build_arg_parser()` and dispatch |

## Manual Testing

Run against sample PDFs to verify changes:

```bash
# Full extraction
python job_card_extractor.py samples/example-01.pdf -o output

# Fast mode
python job_card_extractor.py samples/example-01.pdf -o output --fast-mode

# Check output
cat output/*_job_and_operations.json | python -m json.tool
```

Review `output/annotated/` images to verify area detection and barcode recognition visually.

## Contributing

1. Create a feature branch from `main`
2. Make changes to `job_card_extractor.py`
3. Add or update tests in `tests/`
4. Run `pytest tests/` and verify all tests pass
5. Test against sample PDFs manually
6. Submit a Pull Request

### Adding New Extraction Patterns

1. Define the regex pattern in the operation extraction section
2. Add a matching strategy
3. Test with sample documents that match the new pattern
4. Update logging and metadata collection
5. Add a unit test for the new pattern

See [Architecture - Extensibility](architecture.md#extensibility) for details.

---

See also: [Architecture](architecture.md) | [User Guide](user-guide.md) | [API Reference](api-reference.md) | [Troubleshooting](troubleshooting.md)
