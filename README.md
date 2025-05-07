# Job Card Extractor

A Python tool for extracting job numbers and operations from manufacturing job cards (PDF documents) using OCR and barcode detection.

**Version: 1.0.0**

## Overview

The Job Card Extractor is designed to automate the extraction of critical information from manufacturing job cards by:

1. Processing PDF documents containing job information
2. Detecting and reading barcodes (particularly those starting with "J")
3. Performing OCR (Optical Character Recognition) on text regions
4. Identifying job numbers, quantities, delivery dates, and operations
5. Generating structured JSON data from the extracted information

The tool provides both command-line interface and programmatic access, with options to save processing artifacts such as annotated images and intermediate data.

## Features

- **PDF Processing**: Convert multi-page PDFs to images for analysis
- **Barcode Detection**: Identify and decode various barcode formats
- **OCR Text Extraction**: Extract text from document areas with preprocessing for improved accuracy
- **Smart Information Extraction**:
  - Extract job numbers from header areas
  - Extract quantities and delivery dates from document headers
  - Recognize operations with their numbers, names, and associated barcodes
- **Visual Debugging**: Generate annotated images showing detected areas, barcodes, and text
- **Flexible Output Options**: Save results as clean JSON and/or raw extraction data

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
   git clone <repository-url>
   cd pilot03-service-jobcard-extractor
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

### Output Structure

When run with an output directory, the tool creates:

```
output_dir/
├── example-01_job_and_operations.json  # Clean extracted data
├── example-01_raw.json                # Raw extraction data
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
      "page": 1
    },
    {
      "op_number": "20",
      "op_name": "ASSEMBLY",
      "op_id": "J123456Q20",
      "page": 1
    }
  ]
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
```

## How It Works

1. **PDF Processing**: The PDF is converted to images, one per page
2. **Area Detection**: Horizontal lines are detected to divide the page into logical areas
3. **Barcode Detection**: Each area is scanned for barcodes
4. **OCR Processing**: Text is extracted from each area using EasyOCR
5. **Information Extraction**:
   - Job numbers are identified, typically from the first page header
   - Quantities are extracted from patterns like "Quantity: 100" or "QTY: 250"
   - Delivery dates are extracted from patterns like "Delivery Date: 15/06/2025" or "Date Required: 10-May-2025"
   - Operations are extracted by identifying patterns like "Operation XX Name"
6. **Result Generation**: Data is compiled into a structured JSON format

## Troubleshooting

- **PDF Conversion Issues**: Ensure poppler is properly installed
- **OCR Quality**: Try different language models or adjust preprocessing parameters
- **Barcode Detection**: Ensure images are clear and have sufficient resolution
- **Memory Issues**: For large documents, consider processing pages individually

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.