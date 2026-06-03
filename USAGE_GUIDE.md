# Job Card Extractor — Usage Guide

CLI tool that extracts job numbers and operations from manufacturing job card
PDFs using OCR and barcode detection.

---

## Requirements

- Python 3.6+
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

**`ModuleNotFoundError` (e.g. `No module named 'pdf2image'`):** the virtual
environment is not active. Run `source ~/.venv/job-card-extractor/bin/activate`
first, then re-run the command.

**`Poppler not found in PATH` / `pdfinfo` missing:** Poppler is not installed.
Reinstall it:

- macOS: `brew install poppler`
- Debian/Ubuntu: `sudo apt-get install poppler-utils`
- Fedora/RHEL: `sudo dnf install poppler-utils`

**First run is slow:** EasyOCR downloads its language models (~100MB+) on first
use. Subsequent runs reuse the cached models.

**Permission denied during install:** the installer uses `sudo` for the system
package step on Linux. Run it as a user with `sudo` access, or install Poppler
manually first.

---

## Uninstallation

Remove the virtual environment, and (optionally) Poppler:

```bash
# Remove the virtual environment
rm -rf ~/.venv/job-card-extractor

# Remove Poppler (optional)
brew uninstall poppler              # macOS
sudo apt-get remove poppler-utils   # Debian/Ubuntu
```

---

## Additional Resources

- Repository: https://github.com/COGNIMANEU/pilot03-service-job-card-extractor
- Issue tracker: https://github.com/COGNIMANEU/pilot03-service-job-card-extractor/issues
- See `README.md` for the full feature list and programmatic use.
