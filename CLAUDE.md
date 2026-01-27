# Job Card Extractor

CLI tool that extracts job numbers and operations from manufacturing job card PDFs using OCR and barcode detection.

## Tech Stack
- Python 3.12
- OpenCV, EasyOCR, PyZbar, pdf2image
- Requires poppler system dependency

## Project Structure
- `job_card_extractor.py` - Main extraction logic (single file)
- `tests/` - Unit tests
- `samples/` - Example PDF files for testing

## Development Commands
```bash
# Activate venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run extractor
python job_card_extractor.py samples/example-01.pdf -o output

# Run tests
pytest tests/
```

## Gotchas
- Poppler must be installed (`brew install poppler` on macOS, `apt-get install poppler-utils` on Linux)
- First run downloads EasyOCR language models (~100MB+)
- Processing is CPU-intensive; use `--fast-mode` for quicker but lower quality results
