# Development

Local setup, testing, and contribution guide for Job Card Extractor.

## Project Structure

```
pilot03-service-job-card-extractor/
├── job_card_extractor.py       # Main application (single-file CLI tool)
├── requirements.txt            # Python dependencies
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

- Python 3.6+
- Poppler (PDF conversion backend)

```bash
# macOS
brew install poppler

# Linux (Ubuntu/Debian)
apt-get install poppler-utils
```

### Installation

```bash
git clone https://github.com/COGNIMANEU/pilot03-service-job-card-extractor
cd pilot03-service-job-card-extractor

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

First run will download EasyOCR language models (~100MB+). Ensure internet connectivity.

### Verify Installation

```bash
# Check version
python job_card_extractor.py --version

# Run on sample file
python job_card_extractor.py samples/example-01.pdf -o output
```

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

## Code Organization

The project is a single-file application (`job_card_extractor.py`) organized into sections:

| Section | Lines | Description |
|---------|-------|-------------|
| `ExtractionLogger` | ~47-280 | Logging system for tracking extraction process |
| Version functions | ~285-304 | `get_version()`, `display_version()` |
| Barcode & OCR | ~308-557 | Detection, preprocessing, OCR with caching |
| PDF processing | ~581-732 | Page processing and parallel PDF extraction |
| Job & operations | ~738-1241 | Extraction logic with pattern matching |
| Main processing | ~1247-1456 | Entry point: `process_pdf_document()` |
| CLI interface | ~1462-1558 | Argument parsing and dispatch |

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
