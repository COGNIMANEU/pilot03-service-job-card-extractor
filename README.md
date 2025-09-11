# Job Card Extractor

A Python tool for extracting job numbers and operations from manufacturing job cards (PDF documents) using OCR and barcode detection.

**Version: 1.1.0**

## Overview

The Job Card Extractor is designed to automate the extraction of critical information from manufacturing job cards by:

1. Processing PDF documents containing job information
2. Detecting and reading barcodes (particularly those starting with "J")
3. Performing OCR (Optical Character Recognition) on text regions
4. Identifying job numbers, quantities, delivery dates, and operations
5. Generating structured JSON data from the extracted information

The tool provides both command-line interface and programmatic access, with options to save processing artifacts such as annotated images and intermediate data.

## Features

- **PDF Processing**: Convert multi-page PDFs to images for analysis with parallel processing support
- **Barcode Detection**: Identify and decode various barcode formats with multiple detection strategies
- **OCR Text Extraction**: Extract text from document areas with advanced preprocessing for improved accuracy
- **Smart Information Extraction**:
  - Extract job numbers from header areas
  - Extract quantities and delivery dates from document headers
  - Recognize operations with their numbers, names, and associated barcodes
- **Visual Debugging**: Generate annotated images showing detected areas, barcodes, and text
- **Flexible Output Options**: Save results as clean JSON and/or raw extraction data
- **Comprehensive Logging**: Unified logging system tracking the entire extraction process
- **Extraction Metadata**: Detailed benchmarking data including performance metrics, confidence scores, and quality statistics
- **Performance Optimization**: OCR caching, parallel processing, and enhanced image preprocessing

## Installation

### Prerequisites

- Python 3.6+
- OpenCV
- NumPy
- EasyOCR
- PyZbar
- pdf2image (requires poppler)

### Setup

1. Clone this repository:
   ```
   git clone https://github.com/COGNIMANEU/pilot03-service-job-card-extractor
   cd pilot03-service-job-card-extractor
   ```

2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Install system dependencies:
   - Poppler (required by pdf2image)
     - macOS: `brew install poppler`
     - Linux: `apt-get install poppler-utils`

## Usage

### Command Line Interface

Check the version of the tool:
```
python job_card_extractor.py --version
```

Process a single PDF file:
```
python job_card_extractor.py example-01.pdf -o example_01
```

Process multiple PDF files:
```
python job_card_extractor.py example-01.pdf example-02.pdf -o output_dir
```

### Command Line Options

- `pdf_files`: One or more PDF files to process
- `-v, --version`: Display version information
- `-o, --output-dir`: Directory to save output files (optional)
- `-l, --lang`: Language codes for OCR (default: en)
- `--no-raw`: Skip saving raw extraction data
- `--no-annotated`: Skip saving annotated debug images
- `--no-parallel`: Disable parallel processing for multi-page documents
- `--fast-mode`: Use faster processing with reduced quality enhancements

### Examples

Save outputs to a specific directory:
```
python job_card_extractor.py example-01.pdf -o example_01
```

Process a PDF with multiple languages:
```
python job_card_extractor.py example-01.pdf -l en fr
```

Process a PDF without saving raw data:
```
python job_card_extractor.py example-01.pdf -o example_01 --no-raw
```

Use fast processing mode for quicker results:
```
python job_card_extractor.py example-01.pdf -o example_01 --fast-mode
```

Disable parallel processing:
```
python job_card_extractor.py example-01.pdf -o example_01 --no-parallel
```

### Output Structure

When run with an output directory, the tool creates:

```
output_dir/
├── example-01_job_and_operations.json  # Clean extracted data with metadata
├── example-01_raw.json                # Raw extraction data
├── extraction_process_YYYYMMDD_HHMMSS.log  # Unified extraction log
└── annotated/                         # Debug images
    ├── page_1_areas.jpg
    └── page_2_areas.jpg
```

## Output Format

### Job and Operations JSON

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
      "extraction_strategy": "direct_match",
      "pattern_matched": "^(\\d+(?:\\.\\d+)?)\\s+(.+?)(?:\\s*(?:Scan|~)|$)"
    },
    {
      "op_number": "20",
      "op_name": "ASSEMBLY",
      "op_id": "J123456Q20",
      "page": 1,
      "confidence": 1.3,
      "extraction_strategy": "same_area_match",
      "pattern_matched": "^(?:Operation\\s+)?(\\d+(?:\\.\\d+)?)\\s*[\\n\\r]+\\s*(.+?)(?:\\n|$)"
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
      "success_rate_percent": 100.0,
      "confidence_scores": {},
      "extraction_strategies": {}
    },
    "quality_metrics": {
      "ocr_confidence_avg": 0.0,
      "barcode_detection_rate": 100.0,
      "pattern_match_success": {}
    }
  }
}
```

## Versioning

This project uses [Semantic Versioning](https://semver.org/). The version format is:

```
MAJOR.MINOR.PATCH
```

- MAJOR version for incompatible API changes
- MINOR version for backwards-compatible functionality additions
- PATCH version for backwards-compatible bug fixes

### Version History

- 1.1.0 (September 11, 2025): Added comprehensive logging system, extraction metadata, performance optimizations, and enhanced barcode detection strategies
- 1.0.0 (May 6, 2025): Initial release with core functionality for job card extraction

## Programmatic Usage

You can use the Job Card Extractor in your Python code:

```python
from job_card_extractor import process_pdf_document, __version__

# Check version
print(f"Job Card Extractor version: {__version__}")

# Process a PDF document
result = process_pdf_document(
    pdf_path='example-01.pdf',
    output_dir='output',
    lang_list=['en'],
    save_raw=True,
    save_annotated=True
)

# Access extracted data
job_number = result['job_number']
quantity = result['quantity']
delivery_date = result['delivery_date']
operations = result['operations']

# Process operations
for op in operations:
    print(f"Operation {op['op_number']}: {op['op_name']}")
    print(f"  Confidence: {op.get('confidence', 'N/A')}")
    print(f"  Strategy: {op.get('extraction_strategy', 'N/A')}")

# Access extraction metadata
metadata = result.get('extraction_metadata', {})
if metadata:
    print(f"Processing time: {metadata['document_info']['processing_time_seconds']:.2f}s")
    print(f"Success rate: {metadata['operation_statistics']['success_rate_percent']:.1f}%")
    print(f"Operations per second: {metadata['extraction_info']['performance_metrics']['operations_per_second']:.2f}")
```

## How It Works

1. **PDF Processing**: The PDF is converted to images, one per page with optional parallel processing
2. **Area Detection**: Horizontal lines are detected to divide the page into logical areas
3. **Barcode Detection**: Each area is scanned for barcodes using multiple detection strategies
4. **OCR Processing**: Text is extracted from each area using EasyOCR with advanced preprocessing and caching
5. **Information Extraction**:
   - Job numbers are identified, typically from the first page header
   - Quantities are extracted from patterns like "Quantity: 100" or "QTY: 250"
   - Delivery dates are extracted from patterns like "Delivery Date: 15/06/2025" or "Date Required: 10-May-2025"
   - Operations are extracted by identifying patterns like "Operation XX Name"
6. **Barcode Association**: Operations are matched with barcodes using multiple strategies:
   - Direct match: Barcode contains operation number
   - Same area match: Barcode found in same document area
   - Proximity match: Barcode found in nearby areas
7. **Logging & Metadata**: Comprehensive logging tracks the entire process with performance metrics
8. **Result Generation**: Data is compiled into structured JSON format with extraction metadata

## Logging and Metadata

### Extraction Logs

The tool generates comprehensive logs for each extraction process:

- **Unified Log File**: Single log file per extraction with timestamped entries
- **Process Tracking**: Logs prefixed with `[MAIN]` for general process information
- **Operation Tracking**: Logs prefixed with `[OP-XX]` for operation-specific details
- **Performance Metrics**: Processing times, success rates, and confidence scores

### Extraction Metadata

Each extraction includes detailed metadata for benchmarking and evaluation:

- **extraction_info**: Version, timestamp, processing settings, performance metrics
- **document_info**: Page count, area count, processing time
- **operation_statistics**: Operation counts, success rates, confidence scores, extraction strategies
- **quality_metrics**: OCR confidence averages, barcode detection rates, pattern matching success

### Barcode Detection Strategies

The tool uses multiple strategies to associate barcodes with operations:

1. **direct_match**: Barcode directly contains the operation number
2. **same_area_match**: Barcode found in the same document area as the operation
3. **same_area_fallback**: First barcode found in the same area when no direct match
4. **proximity_match**: Barcode found in nearby areas (±2 areas)
5. **no_barcode_found**: No barcode could be associated with the operation

## Troubleshooting

- **PDF Conversion Issues**: Ensure poppler is properly installed
- **OCR Quality**: Try different language models or adjust preprocessing parameters
- **Barcode Detection**: Ensure images are clear and have sufficient resolution
- **Memory Issues**: For large documents, consider processing pages individually or use `--no-parallel`
- **Performance Issues**: Use `--fast-mode` for quicker processing with reduced quality
- **Logging Issues**: Check that the output directory has write permissions

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
