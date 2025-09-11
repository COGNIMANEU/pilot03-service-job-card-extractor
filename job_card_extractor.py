#!/usr/bin/env python3
"""
Process Job Document

This script processes job documents (PDFs), extracts areas, OCR text, and barcodes,
and outputs a clean JSON with job number and operations information.

It combines functionality from extract_operations_ocr.py and filter_j_barcodes.py
into a single workflow.
"""

__version__ = '1.1.0'

import os
import cv2
import numpy as np
import json
import re
import argparse
import sys
from pathlib import Path
from pdf2image import convert_from_path
import easyocr
from PIL import Image
import warnings
from pyzbar.pyzbar import decode
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from functools import lru_cache
import time
from typing import List, Dict, Tuple, Optional

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

def detect_barcodes(img_crop, enhance_detection=True):
    """Enhanced barcode detection with multiple preprocessing strategies."""
    if img_crop is None or img_crop.size == 0:
        return [], []
        
    result = []
    all_barcodes = []
    
    # Convert to PIL Image if needed
    if isinstance(img_crop, np.ndarray):
        pil_image = Image.fromarray(img_crop)
    else:
        pil_image = img_crop
    
    # Strategy 1: Direct detection on original image
    barcodes = decode(pil_image)
    all_barcodes.extend(barcodes)
    
    if enhance_detection and len(barcodes) == 0:
        # Strategy 2: Try with grayscale conversion
        if len(img_crop.shape) == 3:
            gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
            barcodes_gray = decode(Image.fromarray(gray))
            all_barcodes.extend(barcodes_gray)
        
        # Strategy 3: Try with enhanced contrast
        try:
            if len(img_crop.shape) == 3:
                gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
            else:
                gray = img_crop.copy()
                
            # Apply CLAHE for better contrast
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Try different thresholding methods
            for thresh_type in [cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV]:
                _, binary = cv2.threshold(enhanced, 0, 255, thresh_type + cv2.THRESH_OTSU)
                barcodes_thresh = decode(Image.fromarray(binary))
                all_barcodes.extend(barcodes_thresh)
                
        except Exception as e:
            print(f"Warning: Error in enhanced barcode detection: {e}")
    
    # Remove duplicates and process results
    seen_barcodes = set()
    for barcode in all_barcodes:
        try:
            decoded_data = barcode.data.decode('utf-8', errors='replace')
            cleaned_barcode = clean_barcode_value(decoded_data)
            
            # Avoid duplicates
            if cleaned_barcode not in seen_barcodes and cleaned_barcode:
                seen_barcodes.add(cleaned_barcode)
                result.append({
                    'type': barcode.type,
                    'barcode': cleaned_barcode,
                    'rect': list(barcode.rect),
                    'confidence': getattr(barcode, 'quality', 100)  # Some barcode libraries provide quality
                })
        except Exception as e:
            print(f"Warning: Error processing barcode: {e}")
            continue
    
    return result, all_barcodes

def preprocess_image_for_ocr(crop, enhance_quality=True):
    """Enhanced preprocessing for better OCR results with multiple quality levels."""
    if crop is None or crop.size == 0:
        return None
        
    try:
        # 1. Enhanced denoising with bilateral filter for better edge preservation
        crop_denoised = cv2.bilateralFilter(crop, 9, 75, 75)
        
        # 2. Convert to grayscale early for better processing
        if len(crop_denoised.shape) == 3:
            crop_gray = cv2.cvtColor(crop_denoised, cv2.COLOR_BGR2GRAY)
        else:
            crop_gray = crop_denoised.copy()
        
        # 3. Contrast enhancement using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        crop_enhanced = clahe.apply(crop_gray)
        
        if enhance_quality:
            # 4. Advanced sharpening with unsharp mask
            gaussian = cv2.GaussianBlur(crop_enhanced, (0, 0), 2.0)
            crop_sharpened = cv2.addWeighted(crop_enhanced, 1.5, gaussian, -0.5, 0)
            
            # 5. Morphological operations to clean up text
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            crop_morph = cv2.morphologyEx(crop_sharpened, cv2.MORPH_CLOSE, kernel)
        else:
            crop_morph = crop_enhanced
        
        # 6. Adaptive thresholding with optimized parameters
        crop_bin = cv2.adaptiveThreshold(
            crop_morph, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 15, 10
        )
        
        # 7. Intelligent upscaling based on text density
        min_height = 400 if enhance_quality else 300
        if crop_bin.shape[0] < min_height:
            scale = min_height / crop_bin.shape[0]
            # Use INTER_LANCZOS4 for better text quality
            crop_bin = cv2.resize(
                crop_bin, None, fx=scale, fy=scale, 
                interpolation=cv2.INTER_LANCZOS4
            )
        
        # 8. Convert back to 3 channels for EasyOCR
        return cv2.cvtColor(crop_bin, cv2.COLOR_GRAY2RGB)
        
    except Exception as e:
        print(f"Warning: Error in image preprocessing: {e}")
        # Fallback to basic processing
        if len(crop.shape) == 3:
            crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            crop_gray = crop.copy()
        return cv2.cvtColor(crop_gray, cv2.COLOR_GRAY2RGB)

@lru_cache(maxsize=128)
def _cached_ocr_hash(image_hash: str, reader_id: str) -> str:
    """Cache OCR results based on image hash."""
    return f"{image_hash}_{reader_id}"

# Global cache for OCR results
_ocr_cache = {}

def perform_ocr(reader, image, use_cache=True):
    """Enhanced OCR with caching and confidence scoring."""
    if image is None:
        return ""
        
    try:
        # Generate hash for caching
        if use_cache:
            image_bytes = cv2.imencode('.jpg', image)[1].tobytes()
            image_hash = hashlib.md5(image_bytes).hexdigest()
            reader_id = str(id(reader))  # Simple reader identification
            cache_key = f"{image_hash}_{reader_id}"
            
            if cache_key in _ocr_cache:
                return _ocr_cache[cache_key]
        
        # Perform OCR with detailed results for confidence scoring
        ocr_result = reader.readtext(image, detail=True, paragraph=False)
        
        # Filter results by confidence and clean text
        filtered_lines = []
        for (bbox, text, confidence) in ocr_result:
            # Only include text with reasonable confidence (>0.3)
            if confidence > 0.3 and text.strip():
                cleaned_text = text.strip().replace('_', ' ')
                # Remove obvious OCR artifacts
                if len(cleaned_text) > 1 or cleaned_text.isalnum():
                    filtered_lines.append(cleaned_text)
        
        result = "\n".join(filtered_lines)
        
        # Cache the result
        if use_cache:
            _ocr_cache[cache_key] = result
            # Limit cache size
            if len(_ocr_cache) > 200:
                # Remove oldest entries (simple FIFO)
                oldest_keys = list(_ocr_cache.keys())[:50]
                for key in oldest_keys:
                    del _ocr_cache[key]
        
        return result
        
    except Exception as e:
        print(f"Warning: Error in OCR processing: {e}")
        return ""

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

def process_page(page_num, img, reader, create_debug=True, enhance_quality=True):
    """Optimized page processing with reduced redundancy."""
    start_time = time.time()
    
    try:
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        lines_y = detect_horizontal_lines(img_cv)
        print(f"Page {page_num+1}: Detected {len(lines_y)-2} areas between horizontal lines")

        # For debug visualization (only if needed)
        barcode_annots = [] if create_debug else None
        ocr_annots = [] if create_debug else None
        areas = []

        # Process each area between lines
        for i in range(len(lines_y) - 1):
            y1, y2 = lines_y[i], lines_y[i + 1]
            if y2 - y1 < 50:  # Skip areas that are too small
                continue

            crop = img_cv[y1:y2, :]
            if crop.size == 0:
                continue

            # Enhanced barcode detection
            barcodes_data, raw_barcodes = detect_barcodes(crop, enhance_detection=enhance_quality)
            
            # Collect barcode annotations for debug (only if needed)
            if create_debug and barcode_annots is not None:
                for barcode in raw_barcodes:
                    try:
                        x, y, w, h = barcode.rect
                        abs_rect = (x, y1 + y, w, h)
                        decoded_data = barcode.data.decode('utf-8', errors='replace')
                        barcode_annots.append((abs_rect, clean_barcode_value(decoded_data)))
                    except Exception as e:
                        print(f"Warning: Error processing barcode annotation: {e}")

            # Enhanced OCR processing (single pass)
            crop_for_ocr = preprocess_image_for_ocr(crop, enhance_quality=enhance_quality)
            if crop_for_ocr is not None:
                ocr_text = perform_ocr(reader, crop_for_ocr, use_cache=True)
            else:
                ocr_text = ""

            # For debug annotations (simplified preview)
            if create_debug and ocr_annots is not None:
                preview_text = ocr_text[:80] + "..." if len(ocr_text) > 80 else ocr_text
                ocr_annots.append((y1, preview_text.replace('\n', ' ')))

            # Create area data
            areas.append({
                "page": page_num + 1,
                "area_index": i,
                "bbox": [int(y1), int(y2)],
                "ocr_text": ocr_text,
                "barcodes": barcodes_data
            })

        # Create debug image only if requested
        debug_img = None
        if create_debug:
            debug_img = create_debug_image(img_cv, lines_y, barcode_annots or [], ocr_annots or [])

        processing_time = time.time() - start_time
        print(f"Page {page_num+1}: Processed {len(areas)} areas in {processing_time:.2f}s")
        
        return areas, debug_img
        
    except Exception as e:
        print(f"Error processing page {page_num+1}: {e}")
        return [], None

def extract_areas_from_pdf(pdf_path, lang_list=None, output_dir=None, parallel_processing=True, enhance_quality=True):
    """Optimized PDF extraction with optional parallel processing."""
    if lang_list is None:
        lang_list = ['en']
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    start_time = time.time()
    print(f"Starting PDF processing: {pdf_path}")
    
    # Setup
    images = convert_from_path(pdf_path)
    print(f"Converted PDF to {len(images)} images")
    
    # Create a single OCR reader instance (reuse for better performance)
    reader = easyocr.Reader(lang_list)
    all_areas = []
    debug_images = []
    
    create_debug = output_dir is not None
    
    if parallel_processing and len(images) > 1:
        # Parallel processing for multi-page documents
        print(f"Using parallel processing for {len(images)} pages")
        
        def process_single_page(page_data):
            page_num, img = page_data
            return process_page(page_num, img, reader, create_debug=create_debug, enhance_quality=enhance_quality)
        
        # Use ThreadPoolExecutor for I/O bound OCR operations
        max_workers = min(4, len(images))  # Limit to avoid memory issues
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all page processing tasks
            future_to_page = {executor.submit(process_single_page, (i, img)): i 
                            for i, img in enumerate(images)}
            
            # Collect results in order
            page_results = [None] * len(images)
            for future in as_completed(future_to_page):
                page_num = future_to_page[future]
                try:
                    page_areas, debug_img = future.result()
                    page_results[page_num] = (page_areas, debug_img)
                except Exception as e:
                    print(f"Error processing page {page_num + 1}: {e}")
                    page_results[page_num] = ([], None)
            
            # Flatten results
            for page_areas, debug_img in page_results:
                if page_areas:
                    all_areas.extend(page_areas)
                if debug_img is not None:
                    debug_images.append(debug_img)
    else:
        # Sequential processing
        print("Using sequential processing")
        for page_num, img in enumerate(images):
            page_areas, debug_img = process_page(
                page_num, img, reader, 
                create_debug=create_debug, 
                enhance_quality=enhance_quality
            )
            all_areas.extend(page_areas)
            if debug_img is not None:
                debug_images.append(debug_img)

    # Save debug images if output_dir provided
    if output_dir and debug_images:
        os.makedirs(output_dir, exist_ok=True)
        for page_num, debug_img in enumerate(debug_images):
            if debug_img is not None:
                debug_img_path = os.path.join(output_dir, f'page_{page_num+1}_areas.jpg')
                cv2.imwrite(debug_img_path, debug_img)
                print(f"Saved debug image: {debug_img_path}")
    
    processing_time = time.time() - start_time
    print(f"PDF processing completed in {processing_time:.2f}s - extracted {len(all_areas)} areas")
    
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

def extract_job_details(json_data):
    """
    Enhanced job details extraction with improved pattern matching and validation.

    Args:
        json_data (list): List of area dictionaries from the JSON file

    Returns:
        dict: A dictionary containing job_number, quantity, and delivery_date
    """
    # Initialize result dictionary
    job_details = {
        "job_number": "",
        "quantity": "",
        "delivery_date": ""
    }

    if not json_data:
        return job_details

    # Sort areas by page and area_index to ensure proper ordering
    sorted_areas = sorted(json_data, key=lambda x: (x.get('page', 0), x.get('area_index', 0)))

    # Get first page areas
    first_page_areas = [area for area in sorted_areas if area.get('page', 0) == 1]
    if not first_page_areas:
        return job_details

    # Enhanced job number extraction with multiple strategies
    job_number_patterns = [
        r'(?:Job\s*No\.?|Job\s*Number)[:\s]*([A-Z0-9]+)',
        r'(?:Job)[:\s]*([A-Z0-9]{6,})',  # Job codes are typically 6+ characters
        r'(?:Work\s*Order|WO)[:\s]*([A-Z0-9]+)',
    ]
    
    # Strategy 1: Look for job number in areas with "Job No" text and barcodes
    for area in first_page_areas:
        ocr_text = area.get('ocr_text', '').strip()
        if any(keyword in ocr_text.upper() for keyword in ['JOB NO', 'JOB NUMBER', 'WORK ORDER']):
            # Check if there's a barcode in this area
            if 'barcodes' in area and area['barcodes']:
                barcode_value = area['barcodes'][0].get('barcode', '')
                if len(barcode_value) >= 6:  # Valid job numbers are typically longer
                    job_details["job_number"] = barcode_value
                    break
            
            # Try to extract from OCR text using patterns
            for pattern in job_number_patterns:
                match = re.search(pattern, ocr_text, re.IGNORECASE)
                if match and len(match.group(1)) >= 6:
                    job_details["job_number"] = match.group(1)
                    break
            if job_details["job_number"]:
                break

    # Strategy 2: If no job number found, look for the first substantial barcode
    if not job_details["job_number"]:
        for area in first_page_areas:
            if 'barcodes' in area and area['barcodes']:
                barcode_value = area['barcodes'][0].get('barcode', '')
                # Filter out obviously non-job-number barcodes
                if len(barcode_value) >= 6 and not barcode_value.isdigit():
                    job_details["job_number"] = barcode_value
                    break

    # Find operation boundary for header area detection
    first_op_index = -1
    operation_keywords = ['operation', 'scan barcodes to start', 'op ', 'step ']
    
    for i, area in enumerate(first_page_areas):
        ocr_text = area.get('ocr_text', '').strip().lower()
        if any(keyword in ocr_text for keyword in operation_keywords):
            # Additional check for operation numbers
            if re.search(r'(?:operation|op)\s*\d+', ocr_text) or 'scan barcodes' in ocr_text:
                first_op_index = i
                break

    # Define header areas (before operations)
    header_areas = first_page_areas[:first_op_index] if first_op_index > 0 else first_page_areas

    # Enhanced quantity extraction with better patterns
    quantity_patterns = [
        r'(?:Quantity|QTY|Qty)\s*[:\-]?\s*(\d+(?:\.\d+)?)',  # Basic quantity patterns
        r'(?:Qty\s*of\s*traceable\s*items?)\s*[:\-]?\s*(\d+(?:\.\d+)?)',  # Traceable items
        r'(?:Total\s*Qty?)\s*[:\-]?\s*(\d+(?:\.\d+)?)',  # Total quantity
        r'(?:Pieces?|Pcs?)\s*[:\-]?\s*(\d+(?:\.\d+)?)',  # Pieces
        r'(?:Units?)\s*[:\-]?\s*(\d+(?:\.\d+)?)',  # Units
    ]

    for area in header_areas:
        ocr_text = area.get('ocr_text', '').strip()
        for pattern in quantity_patterns:
            quantity_match = re.search(pattern, ocr_text, re.IGNORECASE)
            if quantity_match:
                qty_value = quantity_match.group(1)
                # Validate quantity (should be reasonable)
                try:
                    qty_float = float(qty_value)
                    if 0 < qty_float <= 10000:  # Reasonable range
                        job_details["quantity"] = qty_value
                        break
                except ValueError:
                    continue
        if job_details["quantity"]:
            break

    # Enhanced delivery date extraction with more formats
    date_patterns = [
        # Standard formats
        r'(?:Delivery\s*Date|Del\.?\s*Date|Due\s*Date|Date\s*Required)\s*[:\-]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        r'(?:Delivery\s*Date|Del\.?\s*Date|Due\s*Date|Date\s*Required)\s*[:\-]?\s*(\d{1,2}[-]\d{1,2}[-]\d{4})',
        # Month name formats
        r'(?:Delivery\s*Date|Del\.?\s*Date|Due\s*Date|Date\s*Required)\s*[:\-]?\s*(\d{1,2}[-\s][A-Za-z]{3,9}[-\s]\d{4})',
        # ISO format
        r'(?:Delivery\s*Date|Del\.?\s*Date|Due\s*Date|Date\s*Required)\s*[:\-]?\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2})',
        # Flexible date patterns
        r'(?:Required\s*by|Needed\s*by|Complete\s*by)\s*[:\-]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
    ]

    for area in header_areas:
        ocr_text = area.get('ocr_text', '').strip()
        for pattern in date_patterns:
            date_match = re.search(pattern, ocr_text, re.IGNORECASE)
            if date_match:
                date_value = date_match.group(1)
                # Basic date validation
                if len(date_value) >= 8:  # Minimum reasonable date length
                    job_details["delivery_date"] = date_value
                    break
        if job_details["delivery_date"]:
            break

    return job_details

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
    Enhanced operations extraction with improved pattern matching and validation.

    Each operation contains:
    - op_number: The number at the beginning of the OCR text
    - op_name: The text following the op_number
    - op_id: The value of the barcode (if available)

    Args:
        json_data (list): List of area dictionaries from the JSON file

    Returns:
        list: List of operation dictionaries
    """
    if not json_data:
        return []
        
    operations_dict = {}  # Dictionary keyed by operation number
    barcodes_by_op_number = {}  # Dictionary to store barcodes
    area_barcodes = {}  # Store barcodes by area for proximity matching

    # Define valid operation number range - allow common manufacturing operation numbers
    MAX_OP_NUMBER = 1000
    MIN_OP_NUMBER = 1  # Allow operations starting from 1, but with better filtering

    try:
        # First pass: Extract operations and collect barcodes
        for area_idx, area in enumerate(json_data):
            ocr_text = area.get('ocr_text', '').strip()
            barcodes = area.get('barcodes', [])
            page = area.get('page', 0)

            if not ocr_text:
                continue

            # Enhanced operation patterns - balanced to catch real operations
            operation_patterns = [
                # Multi-line pattern: operation number on one line, name on next
                r'^(?:Operation\s+)?(\d+(?:\.\d+)?)\s*[\n\r]+\s*(.+?)(?:\n|$)',
                # Single line with "Operation" prefix
                r'^Operation\s+(\d+(?:\.\d+)?)\s+(.+?)(?:\s*(?:Scan|~)|$)',
                # Operation with year pattern (like "150 2022 3D PRINTING")
                r'^(\d+(?:\.\d+)?)\s+(?:20\d\d\s+)?(.+?)(?:\s*(?:Scan|~)|$)',
                # Line-by-line pattern for operations split across lines
                r'(?:^|\n)(\d+(?:\.\d+)?)\s*\n(?:20\d\d\s*\n)?(.+?)(?=\n|$)',
            ]

            # Try each pattern
            for pattern in operation_patterns:
                matches = re.finditer(pattern, ocr_text, re.MULTILINE | re.DOTALL)
                for match in matches:
                    op_number = match.group(1)
                    op_name_raw = match.group(2).strip()

                    # Validate operation number
                    try:
                        op_num_float = float(op_number)
                        op_num_int = int(op_num_float)
                        if not (MIN_OP_NUMBER <= op_num_int <= MAX_OP_NUMBER):
                            continue
                    except (ValueError, TypeError):
                        continue

                    # Clean operation name
                    op_name = clean_operation_name(op_name_raw)
                    
                    # Enhanced filtering to exclude non-operation content
                    if len(op_name) < 2 or op_name.isdigit():
                        continue
                    
                    # Skip obvious non-operations (dates, codes, quantities, etc.)
                    skip_patterns = [
                        r'^\d{1,2}[-/]\w+[-/]\d{4}$',  # Dates like "16-January-2025"
                        r'^[A-Z]{2,3}\d{4,6}$',        # Codes like "AM0135"
                        r'^\d+\.\d+$',                 # Quantities like "10.00"
                        r'^(SCAN|Enter|Activity|Qty|delivered|so|far)\b',  # Common header words
                        r'^[A-Z]{1,3}\d{1,3}$',        # Short codes (but allow if followed by manufacturing terms)
                        r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\b',  # Month names
                        r'^(Entcr|Acttvity)\b',        # OCR errors of "Enter Activity"
                        r'^\d+\.\d+\s*(Qty|delivered)',  # Quantity-related text
                        r'^(Target|Time)\b',           # Table headers
                    ]
                    
                    if any(re.search(pattern, op_name, re.IGNORECASE) for pattern in skip_patterns):
                        continue
                    
                    # Only accept operations that look like manufacturing processes
                    # Must contain meaningful alphabetic content
                    if not re.search(r'[A-Za-z]{3,}', op_name):
                        continue
                    
                    # Additional check: operation names should contain manufacturing-related keywords
                    # or be all caps (common for operation names)
                    manufacturing_indicators = [
                        r'\b(PRINT|CUT|CLEAN|BLAST|MACHINE|MILL|DRILL|WELD|ASSEMBLE|INSPECT|TEST)\b',
                        r'^[A-Z\s]+$',  # All caps operation names
                        r'\b(Wire|Sonic|Dry|EDM|WASH)\b',  # Common operation words
                        r'\b(3D|ULTRA|Bead)\b',  # Specific manufacturing terms
                    ]
                    
                    # Be more lenient - if it has manufacturing indicators OR looks like an operation name
                    has_manufacturing_terms = any(re.search(pattern, op_name, re.IGNORECASE) for pattern in manufacturing_indicators)
                    looks_like_operation = len(op_name) >= 4 and re.search(r'[A-Z]', op_name) and not op_name.isdigit()
                    
                    if not (has_manufacturing_terms or looks_like_operation):
                        continue

                    # Store operation (avoid duplicates, prefer first occurrence)
                    if op_number not in operations_dict:
                        operations_dict[op_number] = {
                            'op_number': op_number,
                            'op_name': op_name,
                            'op_id': '',
                            'page': page,
                            'area_index': area_idx,
                            'confidence': 1.0  # Base confidence
                        }

            # Process barcodes in this area
            if barcodes:
                area_barcodes[area_idx] = []
                for barcode in barcodes:
                    barcode_value = barcode.get('barcode', '')
                    if not barcode_value:
                        continue
                        
                    area_barcodes[area_idx].append(barcode_value)
                    
                    # Enhanced barcode-to-operation matching
                    barcode_patterns = [
                        r'J\w*Q(\d+)$',  # Standard J...Q### format
                        r'.*Q(\d+)$',    # Any barcode ending with Q###
                        r'.*-(\d+)$',    # Barcodes ending with -###
                        r'.*(\d{2,3})$', # Last 2-3 digits as operation number
                    ]
                    
                    for bc_pattern in barcode_patterns:
                        bc_match = re.search(bc_pattern, barcode_value)
                        if bc_match:
                            extracted_op_num = bc_match.group(1)
                            try:
                                if MIN_OP_NUMBER <= int(extracted_op_num) <= MAX_OP_NUMBER:
                                    barcodes_by_op_number[extracted_op_num] = barcode_value
                                    break
                            except ValueError:
                                continue

        # Second pass: Enhanced barcode assignment
        for op_number, operation in operations_dict.items():
            area_idx = operation['area_index']
            
            # Strategy 1: Direct operation number match in barcode
            if op_number in barcodes_by_op_number:
                operation['op_id'] = barcodes_by_op_number[op_number]
                operation['confidence'] += 0.5
                continue
            
            # Strategy 2: Look for barcodes in the same area
            if area_idx in area_barcodes and area_barcodes[area_idx]:
                # Prefer barcodes that contain the operation number
                for barcode_value in area_barcodes[area_idx]:
                    if op_number in barcode_value:
                        operation['op_id'] = barcode_value
                        operation['confidence'] += 0.3
                        break
                
                # If no match found, use the first barcode in the area
                if not operation['op_id'] and area_barcodes[area_idx]:
                    operation['op_id'] = area_barcodes[area_idx][0]
                    operation['confidence'] += 0.1
            
            # Strategy 3: Look for barcodes in nearby areas (proximity matching)
            if not operation['op_id']:
                for nearby_area_idx in range(max(0, area_idx-2), min(len(json_data), area_idx+3)):
                    if nearby_area_idx in area_barcodes and area_barcodes[nearby_area_idx]:
                        for barcode_value in area_barcodes[nearby_area_idx]:
                            if op_number in barcode_value:
                                operation['op_id'] = barcode_value
                                operation['confidence'] += 0.2
                                break
                        if operation['op_id']:
                            break

        # Convert to sorted list and clean up
        operations_list = []
        for op_number in sorted(operations_dict.keys(), key=lambda x: float(x)):
            op = operations_dict[op_number].copy()
            # Remove internal fields
            op.pop('area_index', None)
            op.pop('confidence', None)
            operations_list.append(op)

        return operations_list
        
    except Exception as e:
        print(f"Error in extract_operations: {e}")
        return []

def extract_job_and_operations(json_data):
    """
    Extract both job details and operations from the JSON data in a single call.

    Args:
        json_data (list): List of area dictionaries from the JSON file

    Returns:
        dict: A dictionary containing job details (job number, quantity, delivery date) and a list of operations
    """
    # Extract job details
    job_details = extract_job_details(json_data)

    # Extract operations
    operations = extract_operations(json_data)

    # Return combined result
    return {
        "job_number": job_details["job_number"],
        "quantity": job_details["quantity"],
        "delivery_date": job_details["delivery_date"],
        "operations": operations
    }

#############################################
# Main Processing Function
#############################################

def process_pdf_document(pdf_path, output_dir=None, lang_list=None, save_raw=True, save_annotated=True, 
                        parallel_processing=True, enhance_quality=True):
    """
    Enhanced PDF processing with improved performance and accuracy.

    This function:
    1. Extracts areas, barcodes, and OCR text from the PDF with enhanced preprocessing
    2. Extracts job number and operations using improved pattern matching
    3. Optionally saves annotated images and JSON data
    4. Supports parallel processing for multi-page documents

    Args:
        pdf_path (str): Path to the PDF file to process
        output_dir (str, optional): Directory to save output files
        lang_list (list, optional): List of language codes for OCR. Defaults to ['en']
        save_raw (bool): Whether to save the raw extraction data as JSON
        save_annotated (bool): Whether to save annotated debug images
        parallel_processing (bool): Whether to use parallel processing for multi-page documents
        enhance_quality (bool): Whether to use enhanced image preprocessing for better accuracy

    Returns:
        dict: A dictionary containing the job number and a list of operations
    
    Raises:
        FileNotFoundError: If the PDF file doesn't exist
        Exception: For other processing errors
    """
    start_time = time.time()
    
    try:
        if lang_list is None:
            lang_list = ['en']

        # Validate input
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        input_path = Path(pdf_path)
        file_stem = input_path.stem
        print(f"Processing document: {file_stem}")

        # Create output directory if specified
        annotated_dir = None
        if output_dir:
            try:
                os.makedirs(output_dir, exist_ok=True)
                if save_annotated:
                    annotated_dir = os.path.join(output_dir, "annotated")
                    os.makedirs(annotated_dir, exist_ok=True)
            except Exception as e:
                print(f"Warning: Could not create output directory: {e}")
                output_dir = None

        # Step 1: Extract areas, OCR text, and barcodes from PDF
        print("Step 1: Extracting areas and performing OCR...")
        try:
            areas, debug_images = extract_areas_from_pdf(
                pdf_path,
                lang_list=lang_list,
                output_dir=annotated_dir,
                parallel_processing=parallel_processing,
                enhance_quality=enhance_quality
            )
            
            if not areas:
                print("Warning: No areas extracted from PDF")
                return {
                    "job_number": "",
                    "quantity": "",
                    "delivery_date": "",
                    "operations": []
                }
                
        except Exception as e:
            print(f"Error during area extraction: {e}")
            raise

        # Step 2: Extract job number and operations from the extracted data
        print("Step 2: Extracting job details and operations...")
        try:
            job_and_operations = extract_job_and_operations(areas)
            
            # Validate results
            if not isinstance(job_and_operations, dict):
                raise ValueError("Invalid job and operations data structure")
                
            # Log extraction results
            job_num = job_and_operations.get('job_number', '')
            ops_count = len(job_and_operations.get('operations', []))
            print(f"Extracted job number: {job_num if job_num else 'Not found'}")
            print(f"Extracted {ops_count} operations")
            
        except Exception as e:
            print(f"Error during job/operations extraction: {e}")
            # Return empty structure on error
            job_and_operations = {
                "job_number": "",
                "quantity": "",
                "delivery_date": "",
                "operations": []
            }

        # Step 3: Save outputs if requested
        if output_dir:
            print("Step 3: Saving output files...")
            try:
                # Save raw extraction data if requested
                if save_raw and areas:
                    raw_json_path = os.path.join(output_dir, f"{file_stem}_raw.json")
                    with open(raw_json_path, 'w', encoding='utf-8') as f:
                        json.dump(areas, f, ensure_ascii=False, indent=2)
                    print(f"Raw extraction data saved to {raw_json_path}")

                # Save clean job and operations data
                clean_json_path = os.path.join(output_dir, f"{file_stem}_job_and_operations.json")
                with open(clean_json_path, 'w', encoding='utf-8') as f:
                    json.dump(job_and_operations, f, ensure_ascii=False, indent=2)
                print(f"Job and operations data saved to {clean_json_path}")
                
            except Exception as e:
                print(f"Warning: Error saving output files: {e}")

        processing_time = time.time() - start_time
        print(f"Document processing completed in {processing_time:.2f}s")
        
        return job_and_operations
        
    except FileNotFoundError:
        raise  # Re-raise file not found errors
    except Exception as e:
        print(f"Critical error processing PDF document: {e}")
        # Return empty structure for any other errors
        return {
            "job_number": "",
            "quantity": "",
            "delivery_date": "",
            "operations": []
        }

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
        "--no-parallel",
        action="store_true",
        help="Disable parallel processing for multi-page documents"
    )
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="Use faster processing with reduced quality enhancements"
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
                save_annotated=not args.no_annotated,
                parallel_processing=not args.no_parallel,
                enhance_quality=not args.fast_mode
            )

            # If no output directory specified, print the result to console
            if not args.output_dir:
                print("\nExtracted job and operations:")
                print(json.dumps(result, indent=2))

        except Exception as e:
            print(f"Error processing {pdf_file}: {str(e)}")

if __name__ == "__main__":
    main()