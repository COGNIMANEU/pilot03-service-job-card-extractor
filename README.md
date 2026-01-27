# Job Card Extractor

Automated extraction of job numbers and operations from manufacturing job card PDFs using OCR and barcode detection.

**Version: 1.1.0** | [License: MIT](LICENSE)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Process a PDF
python job_card_extractor.py example-01.pdf -o output
```

## Features

- **PDF Processing**: Multi-page PDF handling with parallel processing support
- **Barcode Detection**: Multiple detection strategies for robust barcode reading
- **OCR Text Extraction**: Advanced preprocessing and caching for accuracy
- **Job Extraction**: Automatically identifies job numbers, quantities, and delivery dates
- **Operation Recognition**: Extracts operation details with confidence scores
- **Visual Debugging**: Annotated images showing detected areas and barcodes
- **Comprehensive Logging**: Detailed extraction logs with performance metrics
- **Flexible Output**: JSON results with raw data and metadata options

## Installation

### Prerequisites

- Python 3.6+
- Poppler: `brew install poppler` (macOS) or `apt-get install poppler-utils` (Linux)
- Dependencies: See `requirements.txt`

### Setup

```bash
git clone https://github.com/COGNIMANEU/pilot03-service-job-card-extractor
cd pilot03-service-job-card-extractor
pip install -r requirements.txt
```

## Usage

### CLI Examples

```bash
# Basic usage
python job_card_extractor.py input.pdf -o output_dir

# Multi-language OCR
python job_card_extractor.py input.pdf -l en fr

# Fast processing (quality tradeoff)
python job_card_extractor.py input.pdf -o output --fast-mode

# Skip raw data and annotated images
python job_card_extractor.py input.pdf -o output --no-raw --no-annotated
```

For complete CLI reference, see [User Guide](docs/user-guide.md).

### Programmatic Usage

```python
from job_card_extractor import process_pdf_document

result = process_pdf_document(
    pdf_path='input.pdf',
    output_dir='output',
    lang_list=['en'],
    save_raw=True
)

print(f"Job: {result['job_number']}")
for op in result['operations']:
    print(f"  Op {op['op_number']}: {op['op_name']}")
```

For detailed API reference, see [API Documentation](docs/api-reference.md).

## Output

The tool generates:

- **job_and_operations.json**: Structured extraction with metadata
- **raw.json**: Raw extraction data for debugging
- **extraction_process_*.log**: Timestamped extraction logs
- **annotated/**: Debug images with detected regions

See [User Guide](docs/user-guide.md#output-structure) for details.

## Documentation

- [**User Guide**](docs/user-guide.md) — CLI usage, options, and examples
- [**API Reference**](docs/api-reference.md) — Programmatic interface and data format
- [**Architecture**](docs/architecture.md) — Technical design and extraction strategies
- [**Troubleshooting**](docs/troubleshooting.md) — Common issues and solutions

## Versioning

This project uses [Semantic Versioning](https://semver.org/). See [Version History](docs/api-reference.md#version-history) for details.

## Contributing

Contributions welcome. Please submit a Pull Request.

## License

MIT
