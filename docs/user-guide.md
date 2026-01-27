# User Guide

Complete reference for using Job Card Extractor from the command line.

## Installation

### Prerequisites

- Python 3.6+
- Poppler (system dependency for PDF processing)
- Dependencies listed in `requirements.txt`

### System Dependencies

**macOS:**
```bash
brew install poppler
```

**Linux (Ubuntu/Debian):**
```bash
apt-get install poppler-utils
```

**Windows:**
Download from [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases/)

### Python Setup

```bash
git clone https://github.com/COGNIMANEU/pilot03-service-job-card-extractor
cd pilot03-service-job-card-extractor
pip install -r requirements.txt
```

## Command Line Interface

### Basic Syntax

```bash
python job_card_extractor.py [OPTIONS] pdf_files...
```

### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--version` | `-v` | — | Display version and exit |
| `--output-dir` | `-o` | Current dir | Output directory for results |
| `--lang` | `-l` | `en` | OCR language codes (space-separated) |
| `--no-raw` | — | False | Skip saving raw extraction data |
| `--no-annotated` | — | False | Skip saving annotated debug images |
| `--no-parallel` | — | False | Disable parallel page processing |
| `--fast-mode` | — | False | Use faster processing (lower quality) |

### Examples

**Single PDF to output directory:**
```bash
python job_card_extractor.py sample.pdf -o results/
```

**Multiple PDFs:**
```bash
python job_card_extractor.py file1.pdf file2.pdf -o batch_output/
```

**Multi-language OCR:**
```bash
python job_card_extractor.py document.pdf -l en fr de
```

**Fast processing (good for preview):**
```bash
python job_card_extractor.py large_doc.pdf -o output --fast-mode
```

**Skip intermediate files (clean output):**
```bash
python job_card_extractor.py input.pdf -o output --no-raw --no-annotated
```

**Disable parallel processing (lower memory usage):**
```bash
python job_card_extractor.py input.pdf -o output --no-parallel
```

**Check version:**
```bash
python job_card_extractor.py --version
```

## Output Structure

All outputs are created in the specified directory (or current directory if `-o` not specified).

```
output_dir/
├── {filename}_job_and_operations.json    # Main extraction result
├── {filename}_raw.json                   # Raw unprocessed data
├── extraction_process_YYYYMMDD_HHMMSS.log  # Processing log
└── annotated/
    ├── page_1_areas.jpg
    ├── page_2_areas.jpg
    └── ...
```

### Main Output: job_and_operations.json

Contains extracted job data with metadata:

- `job_number` — Extracted job identifier
- `quantity` — Quantity from header
- `delivery_date` — Required delivery date
- `operations` — Array of extracted operations
- `extraction_metadata` — Performance and quality metrics

Each operation includes:
- `op_number` — Operation number
- `op_name` — Operation description
- `op_id` — Composite ID
- `confidence` — Extraction confidence score
- `extraction_strategy` — Method used (direct_match, same_area_match, etc.)

### Annotated Images

Located in `annotated/` subdirectory, showing:
- Detected document areas (blue lines)
- Extracted text regions (green boxes)
- Detected barcodes (red outlines)

Useful for debugging poor extraction results.

### Logs

File: `extraction_process_YYYYMMDD_HHMMSS.log`

Contains timestamped entries for:
- Document processing start/end
- Page conversion progress
- Area detection results
- Barcode detection attempts
- OCR processing details
- Operation extraction steps

## Output Format Details

### JSON Structure Example

```json
{
  "job_number": "J123456",
  "quantity": "100",
  "delivery_date": "15/06/2025",
  "operations": [
    {
      "op_number": "10",
      "op_name": "CUTTING",
      "op_id": "J123456Q10",
      "page": 1,
      "confidence": 1.5,
      "extraction_strategy": "direct_match"
    },
    {
      "op_number": "20",
      "op_name": "ASSEMBLY",
      "op_id": "J123456Q20",
      "page": 1,
      "confidence": 1.3,
      "extraction_strategy": "same_area_match"
    }
  ],
  "extraction_metadata": {
    "extraction_info": {
      "extractor_version": "1.1.0",
      "extraction_timestamp": "2025-09-11T22:56:25.261275",
      "processing_settings": {
        "parallel_processing": true,
        "enhance_quality": true,
        "ocr_languages": ["en"],
        "cache_enabled": true
      },
      "performance_metrics": {
        "avg_time_per_operation": 0.781,
        "operations_per_second": 1.28,
        "areas_per_second": 2.08
      }
    },
    "document_info": {
      "total_pages": 3,
      "total_areas": 26,
      "processing_time_seconds": 12.496
    },
    "operation_statistics": {
      "total_operations_found": 16,
      "operations_with_barcodes": 16,
      "success_rate_percent": 100.0
    },
    "quality_metrics": {
      "barcode_detection_rate": 100.0,
      "ocr_confidence_avg": 0.92
    }
  }
}
```

## Performance Tips

### Processing Large Documents

1. **Use parallel processing** (default):
   - Fastest for multi-page documents
   - Requires more memory

2. **Disable parallel for memory-constrained systems:**
   ```bash
   python job_card_extractor.py large.pdf -o output --no-parallel
   ```

3. **Use fast mode for preview:**
   ```bash
   python job_card_extractor.py large.pdf -o output --fast-mode
   ```

### OCR Optimization

- **English only (fast):** `-l en` (default)
- **Add languages as needed:** `-l en fr de`
- More languages = slower processing

### Reducing Output Size

```bash
python job_card_extractor.py input.pdf -o output --no-annotated
```

Skips annotated images (~500KB-1MB per page).

## Language Support

Use ISO 639-1 language codes. Common examples:

| Code | Language |
|------|----------|
| `en` | English |
| `fr` | French |
| `de` | German |
| `es` | Spanish |
| `it` | Italian |
| `pt` | Portuguese |
| `nl` | Dutch |
| `pl` | Polish |

Example with multiple languages:
```bash
python job_card_extractor.py document.pdf -l en fr de
```

## Extraction Strategies

The tool uses multiple strategies to match operations with barcodes:

1. **direct_match** — Barcode contains operation number (highest confidence)
2. **same_area_match** — Barcode in same document area
3. **same_area_fallback** — First barcode from same area
4. **proximity_match** — Barcode in nearby areas
5. **no_barcode_found** — No associated barcode

See [Architecture](architecture.md) for technical details.
