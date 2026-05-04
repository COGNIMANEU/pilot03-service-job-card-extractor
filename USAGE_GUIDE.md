# Job Card Extractor - Usage Guide

## Quick Install

```bash
curl -sSL https://raw.githubusercontent.com/COGNIMANEU/pilot03-service-job-card-extractor/main/install.sh | bash
```

## Requirements

- Python 3.6+
- Poppler (installed automatically by install.sh)
- macOS, Linux, or Windows (PowerShell)

## Setup

After running the install script:

```bash
# Activate the virtual environment
source ~/.venv/job-card-extractor/bin/activate
```

## Basic Usage

```bash
# Process a PDF file
python job_card_extractor.py samples/example-01.pdf -o output

# With multiple languages
python job_card_extractor.py input.pdf -o output -l en fr

# Fast mode (lower quality, faster processing)
python job_card_extractor.py input.pdf -o output --fast-mode
```

## Output

The tool generates:
- `{filename}_job_and_operations.json` - Main extraction results
- `{filename}_raw.json` - Raw extracted data (optional)
- `annotated/` - Debug images showing detected regions

## Deactivate

```bash
deactivate
```

## Troubleshooting

See [Troubleshooting Guide](docs/troubleshooting.md) for common issues.

## Uninstall

```bash
# Remove virtual environment
rm -rf ~/.venv/job-card-extractor

# Remove poppler (optional)
brew uninstall poppler        # macOS
sudo apt-get remove poppler-utils  # Linux
```