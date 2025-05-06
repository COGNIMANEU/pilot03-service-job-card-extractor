#!/usr/bin/env python3
"""
Process Job Document

This script processes job documents (PDFs), extracts areas, OCR text, and barcodes,
and outputs a clean JSON with job number and operations information.

It combines functionality from extract_operations_ocr.py and filter_j_barcodes.py
into a single workflow.
"""

__version__ = '1.0.0'

import os
import cv2
import numpy as np
import json
import re
import argparse
import sys
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image
import warnings
from pyzbar.pyzbar import decode

# Suppress PyTorch pin_memory warning on MPS devices
warnings.filterwarnings(
    "ignore",
    message="'pin_memory' argument is set as true but not supported on MPS now, then device pinned memory won't be used.",
    category=UserWarning,
    module=r"torch.utils.data.dataloader"
)

#############################################
# Version Functions
#############################################

def get_version():
    """
    Return the current version of the Job Card Extractor.

    Returns:
        str: The version string
    """
    return __version__

def display_version():
    """
    Display version information about the Job Card Extractor.

    Prints the version number and additional information to stdout.
    """
    print(f"Job Card Extractor v{__version__}")
    print("(c) 2025 Montimage")
    print("For more information, see the documentation at:")
    print("https://github.com/COGNIMANEU/pilot03-service-job-card-extractor")

#############################################
# Barcode and OCR Extraction Functions
#############################################

def clean_barcode_value(s):
    """Remove all control and non-alphanumeric characters."""
    return ''.join(c for c in s if c.isalnum())

def detect_horizontal_lines(img_cv):
    """Detect horizontal lines in the image."""
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    # Adaptive thresholding for better binarization
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 35, 15)
    # Morphological kernel: wide and thin for horizontal lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (img_cv.shape[1] // 5, 2))
    detect_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    # Find contours of the lines
    contours, _ = cv2.findContours(detect_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter lines by length (must be at least 60% of image width)
    min_line_length = int(img_cv.shape[1] * 0.6)
    lines_y = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w >= min_line_length:
            lines_y.append(y)

    # Add top and bottom of the page
    lines_y = [0] + sorted(lines_y) + [img_cv.shape[0]]
    # Remove duplicates and sort
    return sorted(list(set(lines_y)))

def detect_barcodes(img_crop):
    """Detect barcodes in the image crop."""
    barcodes = decode(Image.fromarray(img_crop))
    result = []
    for barcode in barcodes:
        decoded_data = barcode.data.decode('utf-8', errors='replace')
        result.append({
            'type': barcode.type,
            'barcode': clean_barcode_value(decoded_data),
            'rect': list(barcode.rect)
        })
    return result, barcodes

def preprocess_image_for_ocr(crop):
    """Preprocess image for better OCR results."""
    # 1. Denoise
    crop = cv2.medianBlur(crop, 3)
    # 2. Sharpen
    kernel_sharpen = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    crop = cv2.filter2D(crop, -1, kernel_sharpen)
    # 3. Convert to grayscale
    crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # 4. Adaptive thresholding
    crop_bin = cv2.adaptiveThreshold(crop_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
    # 5. Upscale if too small
    min_height = 600
    if crop_bin.shape[0] < min_height:
        scale = min_height / crop_bin.shape[0]
        crop_bin = cv2.resize(crop_bin, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    # 6. Convert back to 3 channels for EasyOCR
    return cv2.cvtColor(crop_bin, cv2.COLOR_GRAY2RGB)

def perform_ocr(reader, image):
    """Perform OCR on the image."""
    ocr_result = reader.readtext(image, detail=0, paragraph=True)
    # Clean up text
    return "\n".join([line.strip().replace('_', ' ') for line in ocr_result if line.strip()])

def create_debug_image(img_cv, lines_y, barcode_annots, ocr_annots):
    """Create a debug image with visual annotations."""
    debug_img = img_cv.copy()
    # Draw area rectangles (red)
    for i in range(len(lines_y) - 1):
        y1, y2 = lines_y[i], lines_y[i + 1]
        if y2 - y1 < 50:
            continue
        cv2.rectangle(debug_img, (0, y1), (img_cv.shape[1]-1, y2-1), (0, 0, 255), 2)

    # Draw barcodes (green) and values
    for (x, y, w, h), value in barcode_annots:
        cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(debug_img, value, (x, max(y-10,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2, cv2.LINE_AA)

    # Draw OCR text (blue) for each area
    for y1, ocr_text in ocr_annots:
        if ocr_text:
            cv2.putText(debug_img, ocr_text[:80], (5, y1+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

    return debug_img

def process_page(page_num, img, reader):
    """Process a single page of the PDF."""
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    lines_y = detect_horizontal_lines(img_cv)
    print(f"Page {page_num+1}: Detected horizontal lines at y = {lines_y}")

    # For debug visualization
    barcode_annots = []
    ocr_annots = []
    areas = []

    # Process each area between lines
    for i in range(len(lines_y) - 1):
        y1, y2 = lines_y[i], lines_y[i + 1]
        if y2 - y1 < 50:
            continue

        crop = img_cv[y1:y2, :]

        # For debug visualization
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        crop_bin = cv2.adaptiveThreshold(crop_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
        crop_bin_rgb = cv2.cvtColor(crop_bin, cv2.COLOR_GRAY2RGB)
        ocr_preview = easyocr.Reader(['en']).readtext(crop_bin_rgb, detail=0, paragraph=True)
        ocr_text = " ".join([line.strip().replace('_', ' ') for line in ocr_preview if line.strip()])
        ocr_annots.append((y1, ocr_text))

        # For barcode detection
        barcodes_data, raw_barcodes = detect_barcodes(crop)
        for barcode in raw_barcodes:
            x, y, w, h = barcode.rect
            # Adjust barcode rect to page coordinates
            abs_rect = (x, y1 + y, w, h)
            barcode_annots.append((abs_rect, clean_barcode_value(barcode.data.decode('utf-8', errors='replace'))))

        # Process for actual extraction
        crop_for_ocr = preprocess_image_for_ocr(crop)
        ocr_text = perform_ocr(reader, crop_for_ocr)

        # Create area data
        areas.append({
            "page": page_num + 1,
            "area_index": i,
            "bbox": [int(y1), int(y2)],
            "ocr_text": ocr_text,
            "barcodes": barcodes_data
        })

    # Create and return debug image along with areas
    debug_img = create_debug_image(img_cv, lines_y, barcode_annots, ocr_annots)

    return areas, debug_img

def extract_areas_from_pdf(pdf_path, lang_list=None, output_dir=None):
    """Extract areas from PDF with OCR and barcode detection."""
    if lang_list is None:
        lang_list = ['en']
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # Setup
    images = convert_from_path(pdf_path)
    reader = easyocr.Reader(lang_list)
    all_areas = []
    debug_images = []

    # Process each page
    for page_num, img in enumerate(images):
        page_areas, debug_img = process_page(page_num, img, reader)
        all_areas.extend(page_areas)
        debug_images.append(debug_img)

    # Save debug images if output_dir provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        for page_num, debug_img in enumerate(debug_images):
            debug_img_path = os.path.join(output_dir, f'page_{page_num+1}_areas.jpg')
            cv2.imwrite(debug_img_path, debug_img)

    return all_areas, debug_images

#############################################
# Job Number and Operations Extraction Functions
#############################################

def extract_job_number(json_data):
    """
    Extract job number from the JSON data.

    The job number is the first barcode in the first area of the first page,
    typically in the same area with OCR text containing "Job No" string.

    Args:
        json_data (list): List of area dictionaries from the JSON file

    Returns:
        str: The job number or empty string if not found
    """
    # Sort areas by page and area_index to ensure proper ordering
    sorted_areas = sorted(json_data, key=lambda x: (x.get('page', 0), x.get('area_index', 0)))

    # First approach: Look for areas with "Job No" in OCR text
    for area in sorted_areas:
        if area.get('page', 0) != 1:  # Only look at the first page
            continue

        ocr_text = area.get('ocr_text', '').strip()
        if 'Job No' in ocr_text and 'barcodes' in area and area['barcodes']:
            # Return the value of the first barcode in this area
            return area['barcodes'][0].get('barcode', '')

    # Second approach: Just take the first barcode from the first page if available
    for area in sorted_areas:
        if area.get('page', 0) != 1:  # Only look at the first page
            continue

        if 'barcodes' in area and area['barcodes']:
            return area['barcodes'][0].get('barcode', '')

    # If no barcode found, return empty string
    return ''

def clean_operation_name(op_name):
    """
    Clean up operation name by removing scan barcode instructions and other noise.

    Args:
        op_name (str): Raw operation name to clean

    Returns:
        str: Cleaned operation name
    """
    # Remove year prefixes (like "2022" seen in example-01.json)
    year_pattern = r'^(?:20\d\d\s+)'
    op_name = re.sub(year_pattern, '', op_name)

    # Remove various forms of scan barcode instructions
    patterns = [
        # Standard format: "Scan barcodes to start job operation"
        r'\s*[sS]can\s+barcodes\s+(?:t[o0]\s+|to\s+)?start\s+job\s+operation.*$',

        # Hyphenated format: "~Scan-barcodes-to-start-job operation"
        r'\s*~?[sS]can-barcodes-(?:t[o0]|to)-start-job\s+operation.*$',

        # Other common variations
        r'\s*~?\s*[sS]can.*$',  # Catch any remaining scan instructions
    ]

    cleaned_name = op_name
    for pattern in patterns:
        cleaned_name = re.sub(pattern, '', cleaned_name)

    return cleaned_name.strip()

def extract_operations(json_data):
    """
    Extract operations from JSON data areas.

    Each operation contains:
    - op_number: The number at the beginning of the OCR text (may be preceded by "Operation")
    - op_name: The text following the op_number on the same line or next line
    - op_id: The value of the barcode (if available, otherwise empty)

    Args:
        json_data (list): List of area dictionaries from the JSON file

    Returns:
        list: List of operation dictionaries
    """
    # First pass: Extract all operations by their operation numbers
    operations_dict = {}  # Dictionary keyed by operation number
    barcodes_by_op_number = {}  # Dictionary to store barcodes that might match operations

    # Define valid operation number range
    MAX_OP_NUMBER = 1000  # Set a reasonable upper limit for operation numbers

    # First pass: Extract operations and collect potential barcode matches
    for area in json_data:
        ocr_text = area.get('ocr_text', '').strip()
        barcodes = area.get('barcodes', [])

        if not ocr_text:
            continue

        # Modified approach: First check if entire text starts with an operation number
        # This handles cases where operation number is on one line and name is on the next
        operation_pattern_multiline = r'^(?:Operation\s+)?(\d+(?:\.\d+)?)\s*[\n\s]+(.+)'
        match = re.match(operation_pattern_multiline, ocr_text, re.DOTALL)
        if match:
            op_number = match.group(1)
            # Get the first line after the operation number for the name
            remaining_lines = match.group(2).strip().split('\n')
            op_name = remaining_lines[0].strip()

            # Validate operation number is within reasonable range
            try:
                op_num_int = int(float(op_number))
                if op_num_int <= MAX_OP_NUMBER and op_number not in operations_dict:
                    # Clean up op_name (improved pattern to handle hyphenated scan text)
                    op_name = clean_operation_name(op_name)

                    operations_dict[op_number] = {
                        'op_number': op_number,
                        'op_name': op_name,
                        'op_id': '',  # Will be filled in second pass if barcode is available
                        'page': area.get('page', 0)
                    }
            except ValueError:
                pass

        # Also check line by line (original approach)
        for line in ocr_text.split('\n'):
            line = line.strip()
            # Pattern to match "Operation XX Name" or just "XX Name"
            # Allow for "XXX 2022 Name" pattern seen in example-01.json
            operation_pattern = r'^(?:Operation\s+)?(\d+(?:\.\d+)?)\s+(?:20\d\d\s+)?(.*)$'
            match = re.match(operation_pattern, line)

            if match:
                op_number = match.group(1)
                op_name = match.group(2)

                # Validate operation number is within reasonable range
                try:
                    op_num_int = int(float(op_number))
                    if op_num_int > MAX_OP_NUMBER:
                        continue  # Skip unrealistic operation numbers (like 2022)
                except ValueError:
                    continue

                # Clean up op_name (using the separate function)
                op_name = clean_operation_name(op_name)

                if op_number not in operations_dict:
                    operations_dict[op_number] = {
                        'op_number': op_number,
                        'op_name': op_name,
                        'op_id': '',  # Will be filled in second pass if barcode is available
                        'page': area.get('page', 0)
                    }

        # Extract barcodes and associate them with op_numbers when possible
        if barcodes:
            for barcode in barcodes:
                barcode_value = barcode.get('barcode', '')
                if barcode_value.startswith('J'):
                    # Extract operation number from barcode if possible
                    # e.g., J4440801A0Q120 for operation 120
                    barcode_op_match = re.search(r'Q(\d+)$', barcode_value)
                    if barcode_op_match:
                        barcode_op_num = barcode_op_match.group(1)
                        barcodes_by_op_number[barcode_op_num] = barcode_value

                    # Also store by page for proximity matching
                    page = area.get('page', 0)
                    if page not in barcodes_by_op_number:
                        barcodes_by_op_number[page] = []
                    barcodes_by_op_number[page].append(barcode_value)

    # Second pass: Assign barcodes to operations if available
    for op_number, operation in operations_dict.items():
        # First try to match by operation number in barcode (e.g., J4440801A0Q120 for op 120)
        if op_number in barcodes_by_op_number:
            operation['op_id'] = barcodes_by_op_number[op_number]
            continue

        # Look for any J barcode with the operation number at the end
        for barcode_op_num, barcode_value in barcodes_by_op_number.items():
            if isinstance(barcode_op_num, str) and barcode_op_num.isdigit():
                if op_number == barcode_op_num:
                    operation['op_id'] = barcode_value
                    break

    # Convert dictionary to list, sorted by operation number
    operations_list = []
    for op_number in sorted(operations_dict.keys(), key=lambda x: int(float(x))):
        operations_list.append(operations_dict[op_number])

    return operations_list

def extract_job_and_operations(json_data):
    """
    Extract both job number and operations from the JSON data in a single call.

    Args:
        json_data (list): List of area dictionaries from the JSON file

    Returns:
        dict: A dictionary containing the job number and a list of operations
    """
    # Extract job number
    job_number = extract_job_number(json_data)

    # Extract operations
    operations = extract_operations(json_data)

    # Return combined result
    return {
        "job_number": job_number,
        "operations": operations
    }

#############################################
# Main Processing Function
#############################################

def process_pdf_document(pdf_path, output_dir=None, lang_list=None, save_raw=True, save_annotated=True):
    """
    Process a PDF job document to extract job number and operations.

    This function:
    1. Extracts areas, barcodes, and OCR text from the PDF
    2. Extracts job number and operations from this data
    3. Optionally saves:
       - Annotated images showing areas, barcodes, and OCR text
       - Raw JSON with all extraction data
       - Clean JSON with job number and operations

    Args:
        pdf_path (str): Path to the PDF file to process
        output_dir (str, optional): Directory to save output files. If None, no files are saved
        lang_list (list, optional): List of language codes for OCR. Defaults to ['en']
        save_raw (bool): Whether to save the raw extraction data as JSON
        save_annotated (bool): Whether to save annotated debug images

    Returns:
        dict: A dictionary containing the job number and a list of operations
    """
    if lang_list is None:
        lang_list = ['en']

    input_path = Path(pdf_path)
    file_stem = input_path.stem

    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        annotated_dir = os.path.join(output_dir, "annotated")

    # Step 1: Extract areas, OCR text, and barcodes from PDF
    areas, debug_images = extract_areas_from_pdf(
        pdf_path,
        lang_list=lang_list,
        output_dir=annotated_dir if output_dir and save_annotated else None
    )

    # Step 2: Extract job number and operations from the extracted data
    job_and_operations = extract_job_and_operations(areas)

    # Step 3: Save outputs if requested
    if output_dir:
        # Save raw extraction data if requested
        if save_raw:
            raw_json_path = os.path.join(output_dir, f"{file_stem}_raw.json")
            with open(raw_json_path, 'w', encoding='utf-8') as f:
                json.dump(areas, f, ensure_ascii=False, indent=2)
            print(f"Raw extraction data saved to {raw_json_path}")

        # Save clean job and operations data
        clean_json_path = os.path.join(output_dir, f"{file_stem}_job_and_operations.json")
        with open(clean_json_path, 'w', encoding='utf-8') as f:
            json.dump(job_and_operations, f, ensure_ascii=False, indent=2)
        print(f"Job and operations data saved to {clean_json_path}")

    return job_and_operations

#############################################
# Command Line Interface
#############################################

def main():
    parser = argparse.ArgumentParser(
        description="Process PDF job documents and extract job number and operations"
    )
    parser.add_argument(
        "pdf_files",
        nargs='*',  # Changed from '+' to '*' to allow empty list
        help="Path to the PDF file(s) to process"
    )
    parser.add_argument(
        "-o", "--output-dir",
        help="Directory to save output files"
    )
    parser.add_argument(
        "-l", "--lang",
        nargs='+',
        default=['en'],
        help="Language codes for OCR (default: en)"
    )
    parser.add_argument(
        "--no-raw",
        action="store_true",
        help="Don't save raw extraction data"
    )
    parser.add_argument(
        "--no-annotated",
        action="store_true",
        help="Don't save annotated debug images"
    )
    parser.add_argument(
        "-v", "--version",
        action="store_true",
        help="Display version information"
    )
    args = parser.parse_args()

    if args.version:
        display_version()
        sys.exit(0)

    # Ensure pdf_files is provided if not showing version
    if not args.pdf_files:
        parser.print_help()
        print("\nError: At least one PDF file is required unless using --version.")
        sys.exit(1)

    for pdf_file in args.pdf_files:
        print(f"\nProcessing {pdf_file}...")
        try:
            result = process_pdf_document(
                pdf_file,
                output_dir=args.output_dir,
                lang_list=args.lang,
                save_raw=not args.no_raw,
                save_annotated=not args.no_annotated
            )

            # If no output directory specified, print the result to console
            if not args.output_dir:
                print("\nExtracted job and operations:")
                print(json.dumps(result, indent=2))

        except Exception as e:
            print(f"Error processing {pdf_file}: {str(e)}")

if __name__ == "__main__":
    main()