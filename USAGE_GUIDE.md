# Job Card Extractor — Usage Guide

CLI tool that extracts job numbers and operations from manufacturing job card
PDFs using OCR and barcode detection.

---

## Requirements

- Python 3.13
- Poppler (installed automatically by `install.sh`)
- macOS, Linux, or Windows (PowerShell — see `install.ps1`)

---

## Installation

```bash
curl -sSL https://raw.githubusercontent.com/COGNIMANEU/pilot03-service-job-card-extractor/main/install.sh | bash
```

The installer creates a virtual environment at `~/.venv/job-card-extractor` and
installs the system dependency (Poppler) and the Python packages.

---

## Quick Start

### Activate the virtual environment

```bash
source ~/.venv/job-card-extractor/bin/activate
```

### Verify the installation

```bash
python job_card_extractor.py --version
```

### Basic usage

```bash
# Process a PDF file
python job_card_extractor.py samples/example-01.pdf -o output

# With multiple OCR languages
python job_card_extractor.py input.pdf -o output -l en fr

# Fast mode (lower quality, faster processing)
python job_card_extractor.py input.pdf -o output --fast-mode

# Show all options
python job_card_extractor.py --help
```

---

## Output

The tool generates:

- `{filename}_job_and_operations.json` — main extraction results
- `{filename}_raw.json` — raw extracted data (enable with `--raw`)
- `annotated/` — debug images showing detected regions (suppress with `--no-annotated`)

---

## Troubleshooting

### Common Issues

**`ModuleNotFoundError` (e.g. `No module named 'pypdfium2'`):** the virtual
environment is not active. Run `source ~/.venv/job-card-extractor/bin/activate`
first, then re-run the command.

**PDF conversion fails:** the extractor renders PDFs with `pypdfium2`, a
self-contained wheel — no system Poppler is needed. Reinstall the locked
dependencies:

```bash
pip install --require-hashes -r requirements.lock
python -c "import pypdfium2; print(pypdfium2.__version__)"
```

**First run is slow:** EasyOCR downloads its language models (~100MB+) on first
use. Subsequent runs reuse the cached models.

---

## Uninstallation

Remove the virtual environment:

```bash
rm -rf ~/.venv/job-card-extractor
```

---

## Additional Resources

- Repository: https://github.com/COGNIMANEU/pilot03-service-job-card-extractor
- Issue tracker: https://github.com/COGNIMANEU/pilot03-service-job-card-extractor/issues
- See `README.md` for the full feature list and programmatic use.
