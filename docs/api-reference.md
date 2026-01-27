# API Reference

Programmatic interface for Job Card Extractor.

## Main Function

### `process_pdf_document()`

Processes a PDF file and extracts job card information.

```python
process_pdf_document(
    pdf_path: str,
    output_dir: str = None,
    lang_list: List[str] = ["en"],
    save_raw: bool = True,
    save_annotated: bool = True,
    enable_parallel: bool = True,
    enhance_quality: bool = True
) -> dict
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pdf_path` | str | Required | Path to PDF file to process |
| `output_dir` | str | Current dir | Directory for output files |
| `lang_list` | List[str] | `["en"]` | OCR language codes |
| `save_raw` | bool | `True` | Save raw extraction data |
| `save_annotated` | bool | `True` | Save annotated debug images |
| `enable_parallel` | bool | `True` | Use parallel page processing |
| `enhance_quality` | bool | `True` | Apply image quality enhancements |

**Returns:**

Dictionary with structure:
```python
{
    "job_number": str,
    "quantity": str,
    "delivery_date": str,
    "operations": List[dict],
    "extraction_metadata": dict
}
```

**Raises:**

- `FileNotFoundError` — PDF file not found
- `ValueError` — Invalid parameters
- `RuntimeError` — Processing error (check logs)

**Example:**

```python
from job_card_extractor import process_pdf_document

result = process_pdf_document(
    pdf_path='job_card.pdf',
    output_dir='results/',
    lang_list=['en'],
    save_raw=True
)

print(f"Job: {result['job_number']}")
print(f"Quantity: {result['quantity']}")
print(f"Operations: {len(result['operations'])}")
```

## Return Value Structure

### Top Level

```python
{
    "job_number": "J123456",           # Job identifier
    "quantity": "100",                  # Quantity (string)
    "delivery_date": "15/06/2025",      # Delivery deadline
    "operations": [...],                # Extracted operations
    "extraction_metadata": {...}        # Quality/performance metrics
}
```

### Operations Array

Each operation object:

```python
{
    "op_number": "10",                  # Operation number (e.g., "10", "20.5")
    "op_name": "CUTTING",               # Operation description
    "op_id": "J123456Q10",              # Composite ID
    "page": 1,                          # 1-indexed page number
    "confidence": 1.5,                  # Barcode match confidence
    "extraction_strategy": "direct_match", # Strategy used
    "pattern_matched": "^(\\d+...)...$"    # Regex pattern used
}
```

**Extraction Strategies:**
- `direct_match` — Barcode contains operation number
- `same_area_match` — Barcode in same area
- `same_area_fallback` — First barcode in area
- `proximity_match` — Barcode in nearby area
- `no_barcode_found` — No associated barcode

**Confidence Scores:**
- 1.5 = direct_match (highest)
- 1.3 = same_area_match
- 0.8 = proximity_match
- 0.3 = no_barcode_found

### Extraction Metadata

```python
{
    "extraction_info": {
        "extractor_version": "1.1.0",
        "extraction_timestamp": "2025-09-11T22:56:25.261275",
        "processing_settings": {
            "parallel_processing": True,
            "enhance_quality": True,
            "ocr_languages": ["en"],
            "cache_enabled": True
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
        "confidence_scores": {...},      # Per-operation scores
        "extraction_strategies": {...}   # Strategy distribution
    },
    "quality_metrics": {
        "ocr_confidence_avg": 0.92,
        "barcode_detection_rate": 100.0,
        "pattern_match_success": {...}
    }
}
```

## Version Information

### `__version__`

Module-level constant with current version.

```python
from job_card_extractor import __version__

print(f"Job Card Extractor v{__version__}")
# Output: Job Card Extractor v1.1.0
```

## Usage Examples

### Basic Processing

```python
from job_card_extractor import process_pdf_document

# Process with defaults
result = process_pdf_document('input.pdf')
print(f"Extracted {len(result['operations'])} operations")
```

### With Output Files

```python
# Save all outputs
result = process_pdf_document(
    pdf_path='input.pdf',
    output_dir='output/',
    save_raw=True,
    save_annotated=True
)
```

### Multi-Language OCR

```python
# Process French and English document
result = process_pdf_document(
    pdf_path='document.pdf',
    lang_list=['en', 'fr'],
    output_dir='results/'
)
```

### Performance Optimization

```python
# Fast processing (lower quality)
result = process_pdf_document(
    pdf_path='large_document.pdf',
    output_dir='output/',
    enhance_quality=False  # Skip preprocessing
)
```

### Programmatic Access to Data

```python
result = process_pdf_document('input.pdf', output_dir='output/')

# Extract job info
job_number = result['job_number']
quantity = int(result['quantity'])
delivery_date = result['delivery_date']

# Process operations
for operation in result['operations']:
    op_num = operation['op_number']
    op_name = operation['op_name']
    confidence = operation['confidence']

    print(f"Operation {op_num}: {op_name} (confidence: {confidence})")

# Access metadata
metadata = result['extraction_metadata']
processing_time = metadata['document_info']['processing_time_seconds']
success_rate = metadata['operation_statistics']['success_rate_percent']

print(f"\nProcessing time: {processing_time:.2f}s")
print(f"Success rate: {success_rate:.1f}%")
```

### Error Handling

```python
from job_card_extractor import process_pdf_document

try:
    result = process_pdf_document('input.pdf', output_dir='output/')
except FileNotFoundError:
    print("PDF file not found")
except ValueError as e:
    print(f"Invalid parameters: {e}")
except RuntimeError as e:
    print(f"Processing failed: {e}")
    # Check extraction_process_*.log for details
```

### Batch Processing

```python
import os
from pathlib import Path
from job_card_extractor import process_pdf_document

pdf_dir = Path('pdf_files/')
results = []

for pdf_file in pdf_dir.glob('*.pdf'):
    try:
        result = process_pdf_document(
            pdf_path=str(pdf_file),
            output_dir='batch_output/'
        )
        results.append({
            'file': pdf_file.name,
            'job_number': result['job_number'],
            'operations_count': len(result['operations'])
        })
    except Exception as e:
        print(f"Error processing {pdf_file}: {e}")

# Summary
for r in results:
    print(f"{r['file']}: Job {r['job_number']} with {r['operations_count']} operations")
```

## Output Files

When `output_dir` is specified, the following files are created:

### JSON Files

**{filename}_job_and_operations.json**
- Main extraction result
- Contains job data, operations, and metadata
- Ready for programmatic consumption

**{filename}_raw.json** (if `save_raw=True`)
- Raw unprocessed extraction data
- Useful for debugging and analysis
- Includes intermediate values

### Log Files

**extraction_process_YYYYMMDD_HHMMSS.log**
- Timestamped extraction log
- Contains all processing steps
- Useful for troubleshooting

### Images

**annotated/** (if `save_annotated=True`)
- Subdirectory with debug images
- One per page: `page_N_areas.jpg`
- Shows detected regions and barcodes

## Version History

### 1.1.0 (September 11, 2025)

**New Features:**
- Comprehensive logging system with unified log file
- Extraction metadata with performance metrics
- Enhanced barcode detection with multiple strategies
- OCR result caching for performance
- Advanced image preprocessing options

**Improvements:**
- Better confidence scoring for operations
- Detailed quality metrics in output
- Performance optimizations
- Parallel processing support

### 1.0.0 (May 6, 2025)

Initial release with:
- PDF to image conversion
- Area detection
- Barcode detection
- OCR text extraction
- Job card information parsing
- JSON output

## Changelog

See [GitHub Releases](https://github.com/COGNIMANEU/pilot03-service-job-card-extractor/releases) for detailed changelog.
