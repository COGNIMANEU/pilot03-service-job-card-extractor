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
import pypdfium2
import easyocr
from PIL import Image
import warnings
import zxingcpp
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import threading
import time
import logging
from datetime import datetime

# Suppress PyTorch pin_memory warning on MPS devices
warnings.filterwarnings(
    "ignore",
    message="'pin_memory' argument is set as true but not supported on MPS now, then device pinned memory won't be used.",
    category=UserWarning,
    module=r"torch.utils.data.dataloader"
)

#############################################
# Tunable thresholds (named constants, F-CLEAN-027)
#############################################

MIN_LINE_LENGTH_RATIO = 0.6    # a separator line must span this fraction of page width
MIN_AREA_HEIGHT_PX = 50        # skip slivers between separator lines shorter than this
OCR_MIN_CONFIDENCE = 0.3       # drop OCR fragments below this confidence score
OCR_CACHE_MAX_ENTRIES = 200    # bound the OCR result cache
OCR_CACHE_TRIM_COUNT = 50      # evict this many oldest entries when the cache is full
UPSCALE_MIN_HEIGHT_ENHANCED = 400  # upscale crops below this height (enhanced mode)
UPSCALE_MIN_HEIGHT_FAST = 300      # upscale crops below this height (fast mode)
MIN_JOB_NUMBER_LENGTH = 6      # job numbers are at least this many characters
MAX_REASONABLE_QUANTITY = 10000  # sanity ceiling for extracted quantities
MIN_OP_NUMBER = 1              # smallest accepted operation number
MAX_OP_NUMBER = 1000           # largest accepted operation number
MAX_PARALLEL_WORKERS = 4       # thread-pool size cap for parallel page processing
DEBUG_TEXT_MAX_LEN = 80        # OCR/barcode preview length drawn on debug images
PROXIMITY_AREA_RADIUS = 2      # barcode search window (areas) around an operation

# Operation-number patterns applied to area OCR text.
OPERATION_PATTERNS = [
    # Multi-line pattern: operation number on one line, name on next
    r'^(?:Operation\s+)?(\d+(?:\.\d+)?)\s*[\n\r]+\s*(.+?)(?:\n|$)',
    # Single line with "Operation" prefix
    r'^Operation\s+(\d+(?:\.\d+)?)\s+(.+?)(?:\s*(?:Scan|~)|$)',
    # Operation with year pattern (like "150 2022 3D PRINTING")
    r'^(\d+(?:\.\d+)?)\s+(?:20\d\d\s+)?(.+?)(?:\s*(?:Scan|~)|$)',
    # Line-by-line pattern for operations split across lines
    r'(?:^|\n)(\d+(?:\.\d+)?)\s*\n(?:20\d\d\s*\n)?(.+?)(?=\n|$)',
]

# Patterns that decode the operation number embedded in a barcode value.
BARCODE_OP_PATTERNS = [
    r'J\w*Q(\d+)$',   # Standard J...Q### format
    r'.*Q(\d+)$',     # Any barcode ending with Q###
    r'.*-(\d+)$',     # Barcodes ending with -###
    r'.*(\d{2,3})$',  # Last 2-3 digits as operation number
]

# Operation names matching these are header/table noise, not operations.
OP_NAME_SKIP_PATTERNS = [
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

# Operation names usually contain one of these or look like all-caps labels.
MANUFACTURING_INDICATORS = [
    r'\b(PRINT|CUT|CLEAN|BLAST|MACHINE|MILL|DRILL|WELD|ASSEMBLE|INSPECT|TEST)\b',
    r'^[A-Z\s]+$',  # All caps operation names
    r'\b(Wire|Sonic|Dry|EDM|WASH)\b',  # Common operation words
    r'\b(3D|ULTRA|Bead)\b',  # Specific manufacturing terms
]

# Job-number patterns applied to first-page OCR text.
JOB_NUMBER_PATTERNS = [
    r'(?:Job\s*No\.?|Job\s*Number)[:\s]*([A-Z0-9]+)',
    r'(?:Job)[:\s]*([A-Z0-9]{6,})',  # Job codes are typically 6+ characters
    r'(?:Work\s*Order|WO)[:\s]*([A-Z0-9]+)',
]

# Quantity patterns applied to header-area OCR text.
QUANTITY_PATTERNS = [
    r'(?:Quantity|QTY|Qty)\s*[:\-]?\s*(\d+(?:\.\d+)?)',  # Basic quantity patterns
    r'(?:Qty\s*of\s*traceable\s*items?)\s*[:\-]?\s*(\d+(?:\.\d+)?)',  # Traceable items
    r'(?:Total\s*Qty?)\s*[:\-]?\s*(\d+(?:\.\d+)?)',  # Total quantity
    r'(?:Pieces?|Pcs?)\s*[:\-]?\s*(\d+(?:\.\d+)?)',  # Pieces
    r'(?:Units?)\s*[:\-]?\s*(\d+(?:\.\d+)?)',  # Units
]

# Delivery-date patterns applied to header-area OCR text.
DELIVERY_DATE_PATTERNS = [
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

# OCR keywords marking where the operations section starts on page 1.
OPERATION_BOUNDARY_KEYWORDS = ['operation', 'scan barcodes to start', 'op ', 'step ']


def convert_from_path(pdf_path, dpi=200):
    """Render a PDF file to a list of PIL images via pypdfium2.

    Drop-in replacement for ``pdf2image.convert_from_path`` at its default
    200 dpi; pypdfium2 ships a self-contained wheel, so no system poppler is
    needed (see docs/decisions/ocr-stack-spike.md).
    """
    pdf = pypdfium2.PdfDocument(pdf_path)
    try:
        scale = dpi / 72.0
        return [page.render(scale=scale).to_pil() for page in pdf]
    finally:
        pdf.close()


class _BarcodeResult:
    """pyzbar-shaped result wrapping a zxing-cpp Barcode.

    Keeps the ``data``/``type``/``rect``/``quality`` attribute surface the
    pipeline consumed from pyzbar so downstream code is unchanged.
    """

    __slots__ = ('data', 'type', 'rect', 'quality')

    def __init__(self, barcode):
        self.data = barcode.bytes  # raw decoded bytes, like pyzbar's Decoded.data
        # Normalise to pyzbar-style names (e.g. "CODE128") so recorded
        # barcode types stay stable for consumers.
        self.type = str(barcode.format).upper().replace(' ', '')
        pos = barcode.position
        xs = [pos.top_left.x, pos.top_right.x, pos.bottom_right.x, pos.bottom_left.x]
        ys = [pos.top_left.y, pos.top_right.y, pos.bottom_right.y, pos.bottom_left.y]
        x, y = min(xs), min(ys)
        self.rect = (x, y, max(xs) - x, max(ys) - y)
        self.quality = 100  # zxing-cpp reports no quality score


def decode(image):
    """Decode barcodes from a PIL image via zxing-cpp.

    Drop-in replacement for ``pyzbar.pyzbar.decode``; zxing-cpp ships a
    self-contained wheel, so no system libzbar is needed (see
    docs/decisions/ocr-stack-spike.md). Returns pyzbar-shaped results.
    """
    return [_BarcodeResult(b) for b in zxingcpp.read_barcodes(image)]

#############################################
# Logging System
#############################################

class ExtractionLogger:
    """
    Comprehensive logging system for tracking the extraction process.
    Creates a single unified log file containing all extraction information.
    Also collects metadata for benchmarking and evaluation.
    """
    
    def __init__(self, output_dir: str, job_number: str = "unknown"):
        self.output_dir = output_dir
        self.job_number = job_number
        self.logger = None
        self.start_time = datetime.now()
        self.current_operation = None
        
        # Metadata collection
        self.metadata = {
            "extraction_info": {
                "extractor_version": __version__,
                "extraction_timestamp": self.start_time.isoformat(),
                "processing_settings": {},
                "performance_metrics": {}
            },
            "document_info": {
                "total_pages": 0,
                "total_areas": 0,
                "processing_time_seconds": 0.0
            },
            "operation_statistics": {
                "total_operations_found": 0,
                "operations_with_barcodes": 0,
                "success_rate_percent": 0.0,
                "confidence_scores": {},
                "extraction_strategies": {}
            },
            "quality_metrics": {
                "ocr_confidence_avg": 0.0,
                "barcode_detection_rate": 0.0,
                "pattern_match_success": {}
            }
        }
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup unified logger
        self._setup_logger()
    
    def _setup_logger(self):
        """Setup the unified extraction logger."""
        log_filename = f"extraction_process_{self.start_time.strftime('%Y%m%d_%H%M%S')}.log"
        log_path = os.path.join(self.output_dir, log_filename)
        
        # Create logger
        self.logger = logging.getLogger(f"extraction_process_{id(self)}")
        self.logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Create file handler
        file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(file_handler)
        
        # Log initial information
        self.logger.info("="*80)
        self.logger.info("JOB CARD EXTRACTION PROCESS STARTED")
        self.logger.info("="*80)
        self.logger.info(f"Job Number: {self.job_number}")
        self.logger.info(f"Start Time: {self.start_time}")
        self.logger.info(f"Output Directory: {self.output_dir}")
        self.logger.info("")
    
    def setup_operation_logger(self, operation_number: str, operation_name: str = ""):
        """Setup logging for a specific operation (now logs to unified file)."""
        self.current_operation = operation_number
        
        # Log operation header in unified log
        self.logger.info("-" * 60)
        self.logger.info(f"OPERATION {operation_number}: {operation_name}")
        self.logger.info("-" * 60)
        self.logger.info(f"Operation Number: {operation_number}")
        self.logger.info(f"Operation Name: {operation_name}")
        self.logger.info(f"Job Number: {self.job_number}")
        self.logger.info(f"Extraction Start: {datetime.now()}")
        
        return self.logger
    
    def log_main(self, level: str, message: str):
        """Log a message to the unified log."""
        if self.logger:
            getattr(self.logger, level.lower())(f"[MAIN] {message}")
    
    def log_operation(self, operation_number: str, level: str, message: str):
        """Log a message for a specific operation to the unified log."""
        if self.logger:
            getattr(self.logger, level.lower())(f"[OP-{operation_number}] {message}")
    
    def log_pdf_processing_start(self, pdf_path: str, pages_count: int):
        """Log PDF processing start."""
        self.log_main("info", f"Starting PDF processing: {pdf_path}")
        self.log_main("info", f"Total pages to process: {pages_count}")
    
    def log_page_processing(self, page_num: int, areas_count: int, processing_time: float):
        """Log page processing results."""
        self.log_main("info", f"Page {page_num}: Processed {areas_count} areas in {processing_time:.2f}s")
    
    def log_ocr_result(self, operation_number: str, area_index: int, ocr_text: str, confidence_info: str = ""):
        """Log OCR results for an operation."""
        if self.logger:
            self.logger.info(f"[OP-{operation_number}] OCR Result - Area {area_index}:")
            self.logger.info(f"[OP-{operation_number}]   Text Length: {len(ocr_text)} characters")
            self.logger.info(f"[OP-{operation_number}]   Confidence Info: {confidence_info}")
            self.logger.info(f"[OP-{operation_number}]   OCR Text Preview: {ocr_text[:200]}{'...' if len(ocr_text) > 200 else ''}")
    
    def log_barcode_detection(self, operation_number: str, area_index: int, barcodes: list):
        """Log barcode detection results for an operation."""
        if self.logger:
            self.logger.info(f"[OP-{operation_number}] Barcode Detection - Area {area_index}:")
            self.logger.info(f"[OP-{operation_number}]   Barcodes Found: {len(barcodes)}")
            for i, barcode in enumerate(barcodes):
                self.logger.info(f"[OP-{operation_number}]     Barcode {i+1}: {barcode.get('barcode', 'N/A')} (Type: {barcode.get('type', 'N/A')})")
    
    def log_operation_extraction(self, operation_number: str, op_name: str, op_id: str, confidence: float, page: int):
        """Log successful operation extraction."""
        if self.logger:
            self.logger.info(f"[OP-{operation_number}] OPERATION SUCCESSFULLY EXTRACTED:")
            self.logger.info(f"[OP-{operation_number}]   Operation Number: {operation_number}")
            self.logger.info(f"[OP-{operation_number}]   Operation Name: {op_name}")
            self.logger.info(f"[OP-{operation_number}]   Operation ID (Barcode): {op_id}")
            self.logger.info(f"[OP-{operation_number}]   Confidence Score: {confidence:.2f}")
            self.logger.info(f"[OP-{operation_number}]   Found on Page: {page}")
        
        # Also log to main process
        self.log_main("info", f"Extracted Operation {operation_number}: {op_name} (ID: {op_id})")
    
    def log_operation_patterns(self, operation_number: str, patterns_tried: list, successful_pattern: str = ""):
        """Log pattern matching attempts for operation extraction."""
        if self.logger:
            self.logger.debug(f"[OP-{operation_number}] Pattern Matching Attempts:")
            for i, pattern in enumerate(patterns_tried):
                status = "✓ SUCCESS" if pattern == successful_pattern else "✗ Failed"
                self.logger.debug(f"[OP-{operation_number}]   Pattern {i+1}: {pattern} - {status}")
    
    def log_image_preprocessing(self, operation_number: str, area_index: int, preprocessing_steps: list):
        """Log image preprocessing steps."""
        if self.logger:
            self.logger.debug(f"[OP-{operation_number}] Image Preprocessing - Area {area_index}:")
            for step in preprocessing_steps:
                self.logger.debug(f"[OP-{operation_number}]   - {step}")
    
    def log_extraction_summary(self, total_operations: int, successful_extractions: int, processing_time: float):
        """Log final extraction summary."""
        if self.logger:
            self.logger.info("")
            self.logger.info("="*80)
            self.logger.info("EXTRACTION PROCESS COMPLETED")
            self.logger.info("="*80)
            self.logger.info(f"Total Operations Found: {total_operations}")
            self.logger.info(f"Successful Extractions: {successful_extractions}")
            self.logger.info(f"Success Rate: {(successful_extractions/total_operations*100):.1f}%" if total_operations > 0 else "N/A")
            self.logger.info(f"Total Processing Time: {processing_time:.2f}s")
            self.logger.info(f"End Time: {datetime.now()}")
            self.logger.info("="*80)
    
    def set_processing_settings(self, parallel_processing: bool, enhance_quality: bool, lang_list: list):
        """Set processing settings for metadata."""
        self.metadata["extraction_info"]["processing_settings"] = {
            "parallel_processing": parallel_processing,
            "enhance_quality": enhance_quality,
            "ocr_languages": lang_list,
            "cache_enabled": True
        }
    
    def set_document_info(self, total_pages: int, total_areas: int):
        """Set document information for metadata."""
        self.metadata["document_info"]["total_pages"] = total_pages
        self.metadata["document_info"]["total_areas"] = total_areas
    
    def add_operation_metadata(self, operation_number: str, confidence: float, strategy_used: str, pattern_matched: str):
        """Add metadata for a specific operation."""
        self.metadata["operation_statistics"]["confidence_scores"][operation_number] = confidence
        self.metadata["operation_statistics"]["extraction_strategies"][operation_number] = strategy_used
        self.metadata["quality_metrics"]["pattern_match_success"][operation_number] = pattern_matched
    
    def finalize_metadata(self, total_operations: int, successful_extractions: int, processing_time: float):
        """Finalize metadata with summary statistics."""
        self.metadata["document_info"]["processing_time_seconds"] = processing_time
        self.metadata["operation_statistics"]["total_operations_found"] = total_operations
        self.metadata["operation_statistics"]["operations_with_barcodes"] = successful_extractions
        
        if total_operations > 0:
            success_rate = (successful_extractions / total_operations) * 100
            self.metadata["operation_statistics"]["success_rate_percent"] = round(success_rate, 1)
        
        # Calculate average confidence if we have confidence scores
        confidence_scores = list(self.metadata["operation_statistics"]["confidence_scores"].values())
        if confidence_scores:
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            self.metadata["quality_metrics"]["ocr_confidence_avg"] = round(avg_confidence, 2)
        
        # Calculate barcode detection rate
        if total_operations > 0:
            barcode_rate = (successful_extractions / total_operations) * 100
            self.metadata["quality_metrics"]["barcode_detection_rate"] = round(barcode_rate, 1)
        
        # Add performance metrics
        self.metadata["extraction_info"]["performance_metrics"] = {
            "avg_time_per_operation": round(processing_time / total_operations, 3) if total_operations > 0 else 0,
            "operations_per_second": round(total_operations / processing_time, 2) if processing_time > 0 else 0,
            "areas_per_second": round(self.metadata["document_info"]["total_areas"] / processing_time, 2) if processing_time > 0 else 0
        }
    
    def get_metadata(self):
        """Get the collected metadata."""
        return self.metadata.copy()
    
    def close_all_loggers(self):
        """Close the unified logger and its handlers."""
        if self.logger:
            for handler in self.logger.handlers[:]:
                handler.close()
                self.logger.removeHandler(handler)

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

    # Filter lines by length (must be at least MIN_LINE_LENGTH_RATIO of image width)
    min_line_length = int(img_cv.shape[1] * MIN_LINE_LENGTH_RATIO)
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

def _enhance_crop_for_ocr(crop_enhanced, enhance_quality, preprocessing_steps):
    """Sharpen and clean the crop when enhanced quality is requested."""
    if not enhance_quality:
        preprocessing_steps.append("Skipped advanced sharpening (fast mode)")
        return crop_enhanced

    # Advanced sharpening with unsharp mask
    gaussian = cv2.GaussianBlur(crop_enhanced, (0, 0), 2.0)
    crop_sharpened = cv2.addWeighted(crop_enhanced, 1.5, gaussian, -0.5, 0)
    preprocessing_steps.append("Applied unsharp mask sharpening")

    # Morphological operations to clean up text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    crop_morph = cv2.morphologyEx(crop_sharpened, cv2.MORPH_CLOSE, kernel)
    preprocessing_steps.append("Applied morphological closing")
    return crop_morph


def _upscale_for_ocr(crop_bin, enhance_quality, preprocessing_steps):
    """Upscale small crops so OCR sees a minimum text height."""
    min_height = UPSCALE_MIN_HEIGHT_ENHANCED if enhance_quality else UPSCALE_MIN_HEIGHT_FAST
    if crop_bin.shape[0] >= min_height:
        preprocessing_steps.append("No upscaling needed")
        return crop_bin

    scale = min_height / crop_bin.shape[0]
    # Use INTER_LANCZOS4 for better text quality
    crop_bin = cv2.resize(
        crop_bin, None, fx=scale, fy=scale,
        interpolation=cv2.INTER_LANCZOS4
    )
    preprocessing_steps.append(f"Upscaled image by {scale:.2f}x to {crop_bin.shape[1]}x{crop_bin.shape[0]}")
    return crop_bin


def _ocr_grayscale_fallback(crop):
    """Basic grayscale conversion used when enhanced preprocessing fails."""
    if len(crop.shape) == 3:
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        crop_gray = crop.copy()
    return cv2.cvtColor(crop_gray, cv2.COLOR_GRAY2RGB)


def preprocess_image_for_ocr(crop, enhance_quality=True, logger=None, operation_number=None, area_index=None):
    """Enhanced preprocessing for better OCR results with multiple quality levels."""
    if crop is None or crop.size == 0:
        return None

    preprocessing_steps = []

    try:
        # 1. Enhanced denoising with bilateral filter for better edge preservation
        crop_denoised = cv2.bilateralFilter(crop, 9, 75, 75)
        preprocessing_steps.append("Applied bilateral filter for denoising")

        # 2. Convert to grayscale early for better processing
        if len(crop_denoised.shape) == 3:
            crop_gray = cv2.cvtColor(crop_denoised, cv2.COLOR_BGR2GRAY)
            preprocessing_steps.append("Converted to grayscale")
        else:
            crop_gray = crop_denoised.copy()
            preprocessing_steps.append("Image already in grayscale")

        # 3. Contrast enhancement using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        crop_enhanced = clahe.apply(crop_gray)
        preprocessing_steps.append("Applied CLAHE contrast enhancement")

        # 4-5. Sharpening/morphology (skipped in fast mode)
        crop_morph = _enhance_crop_for_ocr(crop_enhanced, enhance_quality, preprocessing_steps)

        # 6. Adaptive thresholding with optimized parameters
        crop_bin = cv2.adaptiveThreshold(
            crop_morph, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 10
        )
        preprocessing_steps.append("Applied adaptive thresholding")

        # 7. Intelligent upscaling based on text density
        crop_bin = _upscale_for_ocr(crop_bin, enhance_quality, preprocessing_steps)

        # 8. Convert back to 3 channels for EasyOCR
        result = cv2.cvtColor(crop_bin, cv2.COLOR_GRAY2RGB)
        preprocessing_steps.append("Converted to RGB for OCR")

        # Log preprocessing steps if logger is provided
        if logger and operation_number and area_index is not None:
            logger.log_image_preprocessing(operation_number, area_index, preprocessing_steps)

        return result

    except Exception as e:
        error_msg = f"Warning: Error in image preprocessing: {e}"
        print(error_msg)
        preprocessing_steps.append(f"ERROR: {error_msg}")

        # Log error if logger is provided
        if logger and operation_number:
            logger.log_operation(operation_number, "error", error_msg)
            if area_index is not None:
                logger.log_image_preprocessing(operation_number, area_index, preprocessing_steps)

        return _ocr_grayscale_fallback(crop)


# Global cache for OCR results. Page processing runs on a ThreadPoolExecutor,
# so every read/write goes through _ocr_cache_lock (F-BUG-045).
_ocr_cache = {}
_ocr_cache_lock = threading.Lock()
_CACHE_MISS = object()


def _ocr_cache_get(cache_key):
    """Return the cached OCR result for cache_key, or the _CACHE_MISS sentinel."""
    with _ocr_cache_lock:
        return _ocr_cache.get(cache_key, _CACHE_MISS)


def _ocr_cache_put(cache_key, result):
    """Store an OCR result, trimming oldest entries past the size cap."""
    with _ocr_cache_lock:
        _ocr_cache[cache_key] = result
        if len(_ocr_cache) > OCR_CACHE_MAX_ENTRIES:
            # Remove oldest entries (simple FIFO)
            oldest_keys = list(_ocr_cache.keys())[:OCR_CACHE_TRIM_COUNT]
            for key in oldest_keys:
                _ocr_cache.pop(key, None)


def perform_ocr(reader, image, use_cache=True, logger=None, operation_number=None, area_index=None):
    """Enhanced OCR with caching and confidence scoring."""
    if image is None:
        return ""

    try:
        # Generate hash for caching
        cache_key = None
        if use_cache:
            image_bytes = cv2.imencode('.jpg', image)[1].tobytes()
            image_hash = hashlib.md5(image_bytes).hexdigest()
            reader_id = str(id(reader))  # Simple reader identification
            cache_key = f"{image_hash}_{reader_id}"

            cached = _ocr_cache_get(cache_key)
            if cached is not _CACHE_MISS:
                if logger and operation_number:
                    logger.log_operation(operation_number, "debug", f"OCR cache hit for area {area_index}")
                return cached

        # Perform OCR with detailed results for confidence scoring
        ocr_result = reader.readtext(image, detail=True, paragraph=False)

        # Filter results by confidence and clean text
        filtered_lines = []
        confidence_scores = []
        for (bbox, text, confidence) in ocr_result:
            # Only include text with reasonable confidence
            if confidence > OCR_MIN_CONFIDENCE and text.strip():
                cleaned_text = text.strip().replace('_', ' ')
                # Remove obvious OCR artifacts
                if len(cleaned_text) > 1 or cleaned_text.isalnum():
                    filtered_lines.append(cleaned_text)
                    confidence_scores.append(confidence)

        result = "\n".join(filtered_lines)

        # Log OCR results if logger is provided
        if logger and operation_number and area_index is not None:
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
            confidence_info = f"Avg: {avg_confidence:.2f}, Lines: {len(filtered_lines)}, Raw results: {len(ocr_result)}"
            logger.log_ocr_result(operation_number, area_index, result, confidence_info)

        if use_cache:
            _ocr_cache_put(cache_key, result)

        return result

    except Exception as e:
        error_msg = f"Warning: Error in OCR processing: {e}"
        print(error_msg)
        if logger and operation_number:
            logger.log_operation(operation_number, "error", error_msg)
        return ""

def create_debug_image(img_cv, lines_y, barcode_annots, ocr_annots):
    """Create a debug image with visual annotations."""
    debug_img = img_cv.copy()
    # Draw area rectangles (red)
    for i in range(len(lines_y) - 1):
        y1, y2 = lines_y[i], lines_y[i + 1]
        if y2 - y1 < MIN_AREA_HEIGHT_PX:
            continue
        cv2.rectangle(debug_img, (0, y1), (img_cv.shape[1]-1, y2-1), (0, 0, 255), 2)

    # Draw barcodes (green) and values
    for (x, y, w, h), value in barcode_annots:
        cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(debug_img, value, (x, max(y-10,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2, cv2.LINE_AA)

    # Draw OCR text (blue) for each area
    for y1, ocr_text in ocr_annots:
        if ocr_text:
            cv2.putText(debug_img, ocr_text[:DEBUG_TEXT_MAX_LEN], (5, y1+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

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
            if y2 - y1 < MIN_AREA_HEIGHT_PX:  # Skip areas that are too small
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
                preview_text = ocr_text[:DEBUG_TEXT_MAX_LEN] + "..." if len(ocr_text) > DEBUG_TEXT_MAX_LEN else ocr_text
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
        # Surface the failure: the caller records which page failed so the
        # document can be reported as partially/failed instead of silently
        # dropping the page (issue #177 / F-BUG-043).
        print(f"Error processing page {page_num+1}: {e}", file=sys.stderr)
        raise

def _process_pages_parallel(images, reader, create_debug, enhance_quality):
    """Process pages on a thread pool.

    Returns (all_areas, indexed_debug_images, failed_pages) where
    indexed_debug_images is a list of (page_num, image) pairs so files are
    named after their real page number (F-BUG-046).
    """
    print(f"Using parallel processing for {len(images)} pages")
    all_areas = []
    indexed_debug = []
    failed_pages = []

    # Use ThreadPoolExecutor for I/O bound OCR operations
    max_workers = min(MAX_PARALLEL_WORKERS, len(images))  # Limit to avoid memory issues
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all page processing tasks
        future_to_page = {
            executor.submit(
                process_page, i, img, reader,
                create_debug=create_debug, enhance_quality=enhance_quality
            ): i
            for i, img in enumerate(images)
        }

        # Collect results in order
        page_results = [None] * len(images)
        for future in as_completed(future_to_page):
            page_num = future_to_page[future]
            try:
                page_results[page_num] = future.result()
            except Exception as e:
                print(f"Error processing page {page_num + 1}: {e}", file=sys.stderr)
                failed_pages.append(page_num + 1)

        # Flatten results, keeping the source page index for debug naming
        for page_num, result in enumerate(page_results):
            if result is None:
                continue
            page_areas, debug_img = result
            if page_areas:
                all_areas.extend(page_areas)
            if debug_img is not None:
                indexed_debug.append((page_num, debug_img))

    return all_areas, indexed_debug, failed_pages


def _process_pages_sequential(images, reader, create_debug, enhance_quality):
    """Process pages one by one; same return shape as the parallel helper."""
    print("Using sequential processing")
    all_areas = []
    indexed_debug = []
    failed_pages = []

    for page_num, img in enumerate(images):
        try:
            page_areas, debug_img = process_page(
                page_num, img, reader,
                create_debug=create_debug,
                enhance_quality=enhance_quality
            )
        except Exception as e:
            print(f"Error processing page {page_num + 1}: {e}", file=sys.stderr)
            failed_pages.append(page_num + 1)
            continue
        all_areas.extend(page_areas)
        if debug_img is not None:
            indexed_debug.append((page_num, debug_img))

    return all_areas, indexed_debug, failed_pages


def _save_debug_images(output_dir, indexed_debug):
    """Write debug images named after their real 1-indexed page number."""
    if not output_dir or not indexed_debug:
        return
    os.makedirs(output_dir, exist_ok=True)
    for page_num, debug_img in indexed_debug:
        if debug_img is not None:
            debug_img_path = os.path.join(output_dir, f'page_{page_num+1}_areas.jpg')
            cv2.imwrite(debug_img_path, debug_img)
            print(f"Saved debug image: {debug_img_path}")


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
    create_debug = output_dir is not None

    if parallel_processing and len(images) > 1:
        all_areas, indexed_debug, failed_pages = _process_pages_parallel(
            images, reader, create_debug, enhance_quality
        )
    else:
        all_areas, indexed_debug, failed_pages = _process_pages_sequential(
            images, reader, create_debug, enhance_quality
        )

    # Save debug images if output_dir provided
    _save_debug_images(output_dir, indexed_debug)

    # A page that raised is data the extraction silently lost — fail the
    # document and name the pages rather than returning partial results that
    # look complete (issue #177 / F-BUG-043).
    if failed_pages:
        raise RuntimeError(
            f"Failed to process page(s) {sorted(failed_pages)} of "
            f"{len(images)} in {pdf_path}"
        )

    processing_time = time.time() - start_time
    print(f"PDF processing completed in {processing_time:.2f}s - extracted {len(all_areas)} areas")

    return all_areas, [img for _, img in indexed_debug]

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

def _extract_job_number_from_areas(first_page_areas):
    """Extract the job number from first-page areas using barcode/OCR strategies."""
    # Strategy 1: Look for job number in areas with "Job No" text and barcodes
    for area in first_page_areas:
        ocr_text = area.get('ocr_text', '').strip()
        if not any(keyword in ocr_text.upper() for keyword in ['JOB NO', 'JOB NUMBER', 'WORK ORDER']):
            continue

        # Check if there's a barcode in this area
        if 'barcodes' in area and area['barcodes']:
            barcode_value = area['barcodes'][0].get('barcode', '')
            if len(barcode_value) >= MIN_JOB_NUMBER_LENGTH:  # Valid job numbers are typically longer
                return barcode_value

        # Try to extract from OCR text using patterns
        for pattern in JOB_NUMBER_PATTERNS:
            match = re.search(pattern, ocr_text, re.IGNORECASE)
            if match and len(match.group(1)) >= MIN_JOB_NUMBER_LENGTH:
                return match.group(1)

    # Strategy 2: If no job number found, look for the first substantial barcode
    for area in first_page_areas:
        if 'barcodes' in area and area['barcodes']:
            barcode_value = area['barcodes'][0].get('barcode', '')
            # Filter out obviously non-job-number barcodes
            if len(barcode_value) >= MIN_JOB_NUMBER_LENGTH and not barcode_value.isdigit():
                return barcode_value

    return ''


def _find_header_areas(first_page_areas):
    """Return the areas before the operations section (the job-card header)."""
    first_op_index = -1
    for i, area in enumerate(first_page_areas):
        ocr_text = area.get('ocr_text', '').strip().lower()
        if any(keyword in ocr_text for keyword in OPERATION_BOUNDARY_KEYWORDS):
            # Additional check for operation numbers
            if re.search(r'(?:operation|op)\s*\d+', ocr_text) or 'scan barcodes' in ocr_text:
                first_op_index = i
                break

    return first_page_areas[:first_op_index] if first_op_index > 0 else first_page_areas


def _extract_quantity(header_areas):
    """Extract the job quantity from header areas using QUANTITY_PATTERNS."""
    for area in header_areas:
        ocr_text = area.get('ocr_text', '').strip()
        for pattern in QUANTITY_PATTERNS:
            quantity_match = re.search(pattern, ocr_text, re.IGNORECASE)
            if quantity_match:
                qty_value = quantity_match.group(1)
                # Validate quantity (should be reasonable)
                try:
                    qty_float = float(qty_value)
                    if 0 < qty_float <= MAX_REASONABLE_QUANTITY:
                        return qty_value
                except ValueError:
                    continue
    return ''


def _extract_delivery_date(header_areas):
    """Extract the delivery date from header areas using DELIVERY_DATE_PATTERNS."""
    for area in header_areas:
        ocr_text = area.get('ocr_text', '').strip()
        for pattern in DELIVERY_DATE_PATTERNS:
            date_match = re.search(pattern, ocr_text, re.IGNORECASE)
            if date_match:
                date_value = date_match.group(1)
                # Basic date validation
                if len(date_value) >= 8:  # Minimum reasonable date length
                    return date_value
    return ''


def extract_job_details(json_data):
    """
    Enhanced job details extraction with improved pattern matching and validation.

    Args:
        json_data (list): List of area dictionaries from the JSON file

    Returns:
        dict: A dictionary containing job_number, quantity, and delivery_date
    """
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

    job_details["job_number"] = _extract_job_number_from_areas(first_page_areas)

    # Quantity and delivery date only come from the header (before operations)
    header_areas = _find_header_areas(first_page_areas)
    job_details["quantity"] = _extract_quantity(header_areas)
    job_details["delivery_date"] = _extract_delivery_date(header_areas)

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

def _iter_operation_matches(ocr_text):
    """Yield (op_number, raw_name, pattern) candidates found in area OCR text."""
    for pattern in OPERATION_PATTERNS:
        for match in re.finditer(pattern, ocr_text, re.MULTILINE | re.DOTALL):
            yield match.group(1), match.group(2).strip(), pattern


def _valid_operation_name(op_number, op_name_raw):
    """Return the cleaned operation name, or None when the candidate is noise."""
    # Validate operation number
    try:
        op_num_int = int(float(op_number))
        if not (MIN_OP_NUMBER <= op_num_int <= MAX_OP_NUMBER):
            return None
    except (ValueError, TypeError):
        return None

    # Clean operation name
    op_name = clean_operation_name(op_name_raw)

    # Enhanced filtering to exclude non-operation content
    if len(op_name) < 2 or op_name.isdigit():
        return None

    # Skip obvious non-operations (dates, codes, quantities, etc.)
    if any(re.search(pattern, op_name, re.IGNORECASE) for pattern in OP_NAME_SKIP_PATTERNS):
        return None

    # Only accept operations that look like manufacturing processes
    # Must contain meaningful alphabetic content
    if not re.search(r'[A-Za-z]{3,}', op_name):
        return None

    # Be more lenient - if it has manufacturing indicators OR looks like an operation name
    has_manufacturing_terms = any(re.search(pattern, op_name, re.IGNORECASE) for pattern in MANUFACTURING_INDICATORS)
    looks_like_operation = len(op_name) >= 4 and re.search(r'[A-Z]', op_name) and not op_name.isdigit()

    if not (has_manufacturing_terms or looks_like_operation):
        return None

    return op_name


def _barcode_operation_number(barcode_value):
    """Decode the operation number embedded in a barcode, or None.

    Uses BARCODE_OP_PATTERNS so matching is by decoded number equality — a
    barcode ending in "105" decodes to 105, not to operation "10" (F-BUG-044).
    """
    for pattern in BARCODE_OP_PATTERNS:
        match = re.search(pattern, barcode_value)
        if not match:
            continue
        try:
            op_num = int(match.group(1))
        except (ValueError, TypeError):
            continue
        if MIN_OP_NUMBER <= op_num <= MAX_OP_NUMBER:
            return op_num
        # Out of range: fall through to the next pattern, like before.
    return None


def _barcode_matches_operation(barcode_value, op_number):
    """True when the barcode's decoded operation number equals op_number."""
    decoded = _barcode_operation_number(barcode_value)
    if decoded is None:
        return False
    try:
        return float(op_number) == float(decoded)
    except (ValueError, TypeError):
        return False


def _extract_area_operations(area, area_idx, operations_dict, logger):
    """First pass for one area: record valid operations into operations_dict."""
    ocr_text = area.get('ocr_text', '').strip()
    if not ocr_text:
        return

    page = area.get('page', 0)
    patterns_tried = list(OPERATION_PATTERNS)

    for op_number, op_name_raw, pattern in _iter_operation_matches(ocr_text):
        op_name = _valid_operation_name(op_number, op_name_raw)
        if op_name is None:
            continue

        # Store operation (avoid duplicates, prefer first occurrence)
        if op_number not in operations_dict:
            operations_dict[op_number] = {
                'op_number': op_number,
                'op_name': op_name,
                'op_id': '',
                'page': page,
                'area_index': area_idx,
                'confidence': 1.0,  # Base confidence
                'extraction_strategy': '',
                'pattern_matched': pattern
            }

            # Setup operation logger and log extraction
            if logger:
                logger.setup_operation_logger(op_number, op_name)
                logger.log_operation_patterns(op_number, patterns_tried, pattern)
                logger.log_operation(op_number, "info", f"Operation found in area {area_idx} on page {page}")
                logger.log_operation(op_number, "info", f"Raw operation name: '{op_name_raw}'")
                logger.log_operation(op_number, "info", f"Cleaned operation name: '{op_name}'")


def _register_area_barcodes(area, area_idx, operations_dict, area_barcodes, barcodes_by_op_number, logger):
    """Collect barcodes for an area and index them by decoded op number."""
    barcodes = area.get('barcodes', [])
    if not barcodes:
        return

    area_barcodes[area_idx] = []

    # Log barcode detection for any operations found in this area
    area_operations = [op for op in operations_dict.values() if op['area_index'] == area_idx]
    for op in area_operations:
        if logger:
            logger.log_barcode_detection(op['op_number'], area_idx, barcodes)

    for barcode in barcodes:
        barcode_value = barcode.get('barcode', '')
        if not barcode_value:
            continue

        area_barcodes[area_idx].append(barcode_value)

        decoded_op_num = _barcode_operation_number(barcode_value)
        if decoded_op_num is not None:
            barcodes_by_op_number[decoded_op_num] = barcode_value
            # Log barcode-to-operation mapping
            if logger and str(decoded_op_num) in operations_dict:
                logger.log_operation(str(decoded_op_num), "info",
                                     f"Barcode '{barcode_value}' mapped to operation {decoded_op_num}")


def _assign_operation_barcodes(operations_dict, barcodes_by_op_number, area_barcodes, area_count, logger):
    """Second pass: attach a barcode to each operation via three strategies."""
    for op_number, operation in operations_dict.items():
        area_idx = operation['area_index']

        if logger:
            logger.log_operation(op_number, "info", "Starting barcode assignment strategies")

        try:
            op_num_key = float(op_number)
        except (ValueError, TypeError):
            op_num_key = None

        # Strategy 1: Direct operation number match in barcode
        if op_num_key is not None and op_num_key in barcodes_by_op_number:
            operation['op_id'] = barcodes_by_op_number[op_num_key]
            operation['confidence'] += 0.5
            operation['extraction_strategy'] = "direct_match"
            if logger:
                logger.log_operation(op_number, "info", f"Strategy 1 SUCCESS: Direct match - Barcode '{operation['op_id']}'")
        else:
            _assign_area_and_proximity_barcodes(
                operation, op_number, area_idx, area_barcodes, area_count, logger
            )

        # Set default strategy if no barcode found
        if not operation['op_id']:
            operation['extraction_strategy'] = "no_barcode_found"

        # Add metadata to logger
        if logger:
            logger.add_operation_metadata(
                op_number,
                operation['confidence'],
                operation['extraction_strategy'],
                operation['pattern_matched']
            )

            # Log final operation extraction result
            logger.log_operation_extraction(
                op_number,
                operation['op_name'],
                operation['op_id'],
                operation['confidence'],
                operation['page']
            )


def _assign_area_and_proximity_barcodes(operation, op_number, area_idx, area_barcodes, area_count, logger):
    """Strategies 2 and 3: same-area match/fallback, then proximity match."""
    # Strategy 2: Look for barcodes in the same area
    if area_idx in area_barcodes and area_barcodes[area_idx]:
        # Prefer barcodes that decode to this operation number (F-BUG-044:
        # decoded-number equality, not substring containment — "105" != "10")
        for barcode_value in area_barcodes[area_idx]:
            if _barcode_matches_operation(barcode_value, op_number):
                operation['op_id'] = barcode_value
                operation['confidence'] += 0.3
                operation['extraction_strategy'] = "same_area_match"
                if logger:
                    logger.log_operation(op_number, "info", f"Strategy 2 SUCCESS: Same area match - Barcode '{barcode_value}'")
                break

        # Fallback: first barcode in the area that does not decode to a
        # *different* operation number, so unrelated barcodes are not claimed.
        if not operation['op_id']:
            for barcode_value in area_barcodes[area_idx]:
                if _barcode_operation_number(barcode_value) is None:
                    operation['op_id'] = barcode_value
                    operation['confidence'] += 0.1
                    operation['extraction_strategy'] = "same_area_fallback"
                    if logger:
                        logger.log_operation(op_number, "info", f"Strategy 2 FALLBACK: First unclaimed barcode in area - '{barcode_value}'")
                    break

    # Strategy 3: Look for barcodes in nearby areas (proximity matching)
    if not operation['op_id']:
        if logger:
            logger.log_operation(op_number, "info", "Trying Strategy 3: Proximity matching")
        for nearby_area_idx in range(max(0, area_idx - PROXIMITY_AREA_RADIUS),
                                     min(area_count, area_idx + PROXIMITY_AREA_RADIUS + 1)):
            for barcode_value in area_barcodes.get(nearby_area_idx, []):
                if _barcode_matches_operation(barcode_value, op_number):
                    operation['op_id'] = barcode_value
                    operation['confidence'] += 0.2
                    operation['extraction_strategy'] = "proximity_match"
                    if logger:
                        logger.log_operation(op_number, "info", f"Strategy 3 SUCCESS: Nearby area {nearby_area_idx} - Barcode '{barcode_value}'")
                    break
            if operation['op_id']:
                break


def _finalize_operations(operations_dict, logger):
    """Sort operations numerically, strip internals, and log the summary."""
    operations_list = []
    successful_extractions = 0
    for op_number in sorted(operations_dict.keys(), key=lambda x: float(x)):
        op = operations_dict[op_number].copy()
        if op.get('op_id'):
            successful_extractions += 1
        # Remove internal fields but keep metadata for final output
        op.pop('area_index', None)
        # Keep confidence, extraction_strategy, and pattern_matched for metadata
        operations_list.append(op)

    if logger:
        logger.log_main("info", f"Operation extraction completed: {len(operations_list)} operations found, {successful_extractions} with barcodes")

    return operations_list


def extract_operations(json_data, logger=None):
    """
    Enhanced operations extraction with improved pattern matching and validation.

    Each operation contains:
    - op_number: The number at the beginning of the OCR text
    - op_name: The text following the op_number
    - op_id: The value of the barcode (if available)

    Args:
        json_data (list): List of area dictionaries from the JSON file
        logger (ExtractionLogger, optional): Logger instance for tracking extraction

    Returns:
        list: List of operation dictionaries
    """
    if not json_data:
        return []

    operations_dict = {}       # Keyed by operation number string
    barcodes_by_op_number = {} # Decoded op number -> barcode value
    area_barcodes = {}         # Area index -> list of barcode values

    if logger:
        logger.log_main("info", f"Starting operation extraction from {len(json_data)} areas")

    try:
        # First pass: extract operations and collect barcodes per area
        for area_idx, area in enumerate(json_data):
            _extract_area_operations(area, area_idx, operations_dict, logger)
            _register_area_barcodes(area, area_idx, operations_dict,
                                    area_barcodes, barcodes_by_op_number, logger)

        # Second pass: assign barcodes to operations
        _assign_operation_barcodes(operations_dict, barcodes_by_op_number,
                                   area_barcodes, len(json_data), logger)

        return _finalize_operations(operations_dict, logger)

    except Exception as e:
        # Propagate: returning [] would write an empty-operations result that
        # looks like a legitimate extraction (issue #177 / F-BUG-043).
        print(f"Error in extract_operations: {e}", file=sys.stderr)
        raise

def extract_job_and_operations(json_data, logger=None):
    """
    Extract both job details and operations from the JSON data in a single call.

    Args:
        json_data (list): List of area dictionaries from the JSON file
        logger (ExtractionLogger, optional): Logger instance for tracking extraction

    Returns:
        dict: A dictionary containing job details (job number, quantity, delivery date) and a list of operations
    """
    # Extract job details
    job_details = extract_job_details(json_data)
    
    if logger:
        logger.log_main("info", f"Job details extracted - Number: {job_details['job_number']}, Quantity: {job_details['quantity']}, Delivery: {job_details['delivery_date']}")

    # Extract operations
    operations = extract_operations(json_data, logger)

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

def _prepare_output_paths(output_dir, save_annotated):
    """Create output directories; returns (output_dir, annotated_dir) or (None, None)."""
    if not output_dir:
        return None, None
    try:
        os.makedirs(output_dir, exist_ok=True)
        annotated_dir = None
        if save_annotated:
            annotated_dir = os.path.join(output_dir, "annotated")
            os.makedirs(annotated_dir, exist_ok=True)
        return output_dir, annotated_dir
    except Exception as e:
        print(f"Warning: Could not create output directory: {e}")
        return None, None


def _init_extraction_logger(output_dir, pdf_path, lang_list, parallel_processing, enhance_quality):
    """Create the ExtractionLogger, or None when logging cannot be set up."""
    if not output_dir:
        return None
    try:
        logger = ExtractionLogger(output_dir, "unknown")  # Job number will be updated later
        logger.set_processing_settings(parallel_processing, enhance_quality, lang_list)
        logger.log_main("info", f"Processing PDF: {pdf_path}")
        logger.log_main("info", f"Language codes: {lang_list}")
        logger.log_main("info", f"Parallel processing: {parallel_processing}")
        logger.log_main("info", f"Enhanced quality: {enhance_quality}")
        return logger
    except Exception as e:
        print(f"Warning: Could not initialize logging system: {e}")
        return None


def _run_area_extraction(pdf_path, lang_list, annotated_dir, parallel_processing, enhance_quality, logger):
    """Step 1: extract areas/OCR/barcodes, propagating failures."""
    print("Step 1: Extracting areas and performing OCR...")
    if logger:
        logger.log_main("info", "Step 1: Starting area extraction and OCR processing")
    try:
        return extract_areas_from_pdf(
            pdf_path,
            lang_list=lang_list,
            output_dir=annotated_dir,
            parallel_processing=parallel_processing,
            enhance_quality=enhance_quality
        )
    except Exception as e:
        error_msg = f"Error during area extraction: {e}"
        print(error_msg)
        if logger:
            logger.log_main("error", error_msg)
        raise


def _record_document_info(logger, pdf_path, areas):
    """Record page/area counts on the logger; never fails the extraction."""
    # Re-render the PDF to get the page count. Diagnostic only — a metadata
    # failure must not fail an otherwise successful extraction.
    try:
        # Module-level lookup: tests patch convert_from_path.
        images = convert_from_path(pdf_path)
        logger.set_document_info(len(images), len(areas))
    except Exception as e:
        logger.log_main("warning", f"Could not record page count: {e}")
        logger.set_document_info(0, len(areas))


def _run_job_extraction(areas, logger, pdf_path):
    """Step 2: extract job details and operations, propagating failures."""
    print("Step 2: Extracting job details and operations...")
    if logger:
        logger.log_main("info", "Step 2: Starting job details and operations extraction")
    try:
        job_and_operations = extract_job_and_operations(areas, logger)

        # Update logger with job number and document info if available
        if logger:
            if job_and_operations.get('job_number'):
                logger.job_number = job_and_operations['job_number']
            _record_document_info(logger, pdf_path, areas)

        # Validate results
        if not isinstance(job_and_operations, dict):
            raise ValueError("Invalid job and operations data structure")

        # Log extraction results
        job_num = job_and_operations.get('job_number', '')
        ops_count = len(job_and_operations.get('operations', []))
        print(f"Extracted job number: {job_num if job_num else 'Not found'}")
        print(f"Extracted {ops_count} operations")
        return job_and_operations
    except Exception as e:
        error_msg = f"Error during job/operations extraction: {e}"
        print(error_msg, file=sys.stderr)
        if logger:
            logger.log_main("error", error_msg)
        # Propagate: an empty result must never be written in place of a
        # failed extraction (issue #177 / F-BUG-043).
        raise


def _attach_extraction_metadata(logger, job_and_operations, start_time):
    """Step 3: finalize logger metadata and attach it to the result."""
    processing_time = time.time() - start_time
    total_operations = len(job_and_operations.get('operations', []))
    successful_extractions = sum(1 for op in job_and_operations.get('operations', []) if op.get('op_id'))

    logger.finalize_metadata(total_operations, successful_extractions, processing_time)
    logger.log_main("info", f"Processing completed in {processing_time:.2f} seconds")
    logger.log_main("info", f"Total operations: {total_operations}, Successful extractions: {successful_extractions}")

    # Add extraction metadata to the final JSON output
    job_and_operations["extraction_metadata"] = logger.get_metadata()


def _save_extraction_outputs(output_dir, file_stem, areas, job_and_operations, save_raw, logger):
    """Step 4: write the raw and clean JSON outputs, propagating failures."""
    print("Step 4: Saving output files...")
    if logger:
        logger.log_main("info", "Step 4: Saving output files")
    try:
        # Save raw extraction data if requested
        if save_raw and areas:
            raw_json_path = os.path.join(output_dir, f"{file_stem}_raw.json")
            with open(raw_json_path, 'w', encoding='utf-8') as f:
                json.dump(areas, f, ensure_ascii=False, indent=2)
            print(f"Raw extraction data saved to {raw_json_path}")
            if logger:
                logger.log_main("info", f"Raw extraction data saved to {raw_json_path}")

        # Save clean job and operations data (now includes metadata)
        clean_json_path = os.path.join(output_dir, f"{file_stem}_job_and_operations.json")
        with open(clean_json_path, 'w', encoding='utf-8') as f:
            json.dump(job_and_operations, f, ensure_ascii=False, indent=2)
        print(f"Job and operations data saved to {clean_json_path}")
        if logger:
            logger.log_main("info", f"Job and operations data saved to {clean_json_path}")
    except Exception as e:
        error_msg = f"Error saving output files: {e}"
        print(error_msg, file=sys.stderr)
        if logger:
            logger.log_main("error", error_msg)
        # A result file that could not be written is a failed
        # extraction, not a successful one (issue #177 / F-BUG-043).
        raise


def _empty_result(logger):
    """Result shape returned when the document yields no areas."""
    return {
        "job_number": "",
        "quantity": "",
        "delivery_date": "",
        "operations": [],
        "extraction_metadata": logger.get_metadata() if logger else {}
    }


def process_pdf_document(pdf_path, output_dir=None, lang_list=None, save_raw=True, save_annotated=True,
                        parallel_processing=True, enhance_quality=True):
    """
    Enhanced PDF processing with improved performance and accuracy.

    Extracts areas/barcodes/OCR text from the PDF, derives job number and
    operations, optionally saves annotated images and JSON data, supports
    parallel processing, and logs the extraction process.

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
    logger = None

    try:
        if lang_list is None:
            lang_list = ['en']

        # Validate input
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        file_stem = Path(pdf_path).stem
        print(f"Processing document: {file_stem}")

        output_dir, annotated_dir = _prepare_output_paths(output_dir, save_annotated)
        logger = _init_extraction_logger(output_dir, pdf_path, lang_list,
                                         parallel_processing, enhance_quality)

        areas, _debug_images = _run_area_extraction(
            pdf_path, lang_list, annotated_dir, parallel_processing, enhance_quality, logger
        )
        if not areas:
            print("Warning: No areas extracted from PDF")
            if logger:
                logger.log_main("warning", "No areas extracted from PDF")
            return _empty_result(logger)

        job_and_operations = _run_job_extraction(areas, logger, pdf_path)

        if logger:
            _attach_extraction_metadata(logger, job_and_operations, start_time)

        if output_dir:
            _save_extraction_outputs(output_dir, file_stem, areas,
                                     job_and_operations, save_raw, logger)

        # Step 5: Finalize and return results
        print("Processing completed successfully!")
        return job_and_operations

    except FileNotFoundError:
        if logger:
            logger.log_main("error", f"PDF file not found: {pdf_path}")
            logger.close_all_loggers()
        raise  # Re-raise file not found errors
    except Exception as e:
        error_msg = f"Critical error processing PDF document: {e}"
        print(error_msg, file=sys.stderr)
        if logger:
            logger.log_main("error", error_msg)
            logger.close_all_loggers()
        # Propagate: callers must see the failure, never an empty result
        # (issue #177 / F-BUG-043).
        raise

#############################################
# Command Line Interface
#############################################

def _build_arg_parser():
    """Build the CLI argument parser.

    ``--raw``/``--no-raw`` and ``--parallel``/``--no-parallel`` share a
    ``dest`` so both directions are real switches: the later flag wins, so a
    ``--no-*`` flag actually changes behavior instead of being a no-op
    (F-CLEAN-026).
    """
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
        "--raw",
        dest="save_raw",
        action="store_true",
        help="Save raw extraction data (default: off)"
    )
    parser.add_argument(
        "--no-raw",
        dest="save_raw",
        action="store_false",
        help="Don't save raw extraction data (overrides --raw)"
    )
    parser.add_argument(
        "--no-annotated",
        action="store_true",
        default=False,
        help="Don't save annotated debug images (default: False - images are saved)"
    )
    parser.add_argument(
        "--parallel",
        dest="parallel",
        action="store_true",
        help="Enable parallel processing for multi-page documents"
    )
    parser.add_argument(
        "--no-parallel",
        dest="parallel",
        action="store_false",
        help="Disable parallel processing for multi-page documents (overrides --parallel)"
    )
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        default=False,
        help="Use faster processing with reduced quality enhancements (default: False)"
    )
    parser.add_argument(
        "-v", "--version",
        action="store_true",
        help="Display version information"
    )
    # Defaults preserve the previous effective behavior: raw output and
    # parallel processing are off unless explicitly enabled.
    parser.set_defaults(save_raw=False, parallel=False)
    return parser


def _process_pdf_file(pdf_file, args):
    """Run the extractor for one PDF and print the result when needed."""
    result = process_pdf_document(
        pdf_file,
        output_dir=args.output_dir,
        lang_list=args.lang,
        save_raw=args.save_raw,
        save_annotated=not args.no_annotated,
        parallel_processing=args.parallel,
        enhance_quality=not args.fast_mode
    )

    # If no output directory specified, print the result to console
    if not args.output_dir:
        print("\nExtracted job and operations:")
        print(json.dumps(result, indent=2))


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.version:
        display_version()
        return 0

    # Ensure pdf_files is provided if not showing version
    if not args.pdf_files:
        parser.print_help()
        print("\nError: At least one PDF file is required unless using --version.")
        return 1

    failed_files = []
    for pdf_file in args.pdf_files:
        print(f"\nProcessing {pdf_file}...")
        try:
            _process_pdf_file(pdf_file, args)
        except Exception as e:
            # Record the failure and keep processing the rest of the batch;
            # the exit code reports it (issue #177 / F-BUG-043).
            print(f"Error processing {pdf_file}: {str(e)}", file=sys.stderr)
            failed_files.append(pdf_file)

    if failed_files:
        print(
            f"Failed to process {len(failed_files)} file(s): "
            f"{', '.join(failed_files)}",
            file=sys.stderr,
        )
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())