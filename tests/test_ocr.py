#!/usr/bin/env python3
import unittest
import sys
import os
import numpy as np
import cv2  # Import cv2 at the module level
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so we can import job_card_extractor
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import job_card_extractor

class TestOCRFunctions(unittest.TestCase):
    @patch('cv2.medianBlur')
    @patch('cv2.filter2D')
    @patch('cv2.cvtColor')
    @patch('cv2.adaptiveThreshold')
    @patch('cv2.resize')
    def test_preprocess_image_for_ocr(self, mock_resize, mock_threshold, mock_cvtcolor,
                                    mock_filter, mock_median):
        """Test the image preprocessing function for OCR"""
        # Create dummy image
        crop = np.zeros((300, 400, 3), dtype=np.uint8)

        # Set up mocks to pass through the data
        mock_median.return_value = crop
        mock_filter.return_value = crop
        mock_cvtcolor.side_effect = [
            np.zeros((300, 400), dtype=np.uint8),  # First call (to grayscale)
            np.zeros((300, 400, 3), dtype=np.uint8)  # Second call (back to RGB)
        ]
        mock_threshold.return_value = np.zeros((300, 400), dtype=np.uint8)
        mock_resize.return_value = np.zeros((600, 800), dtype=np.uint8)

        # Call function
        result = job_card_extractor.preprocess_image_for_ocr(crop)

        # Verify calls
        mock_median.assert_called_once()
        mock_filter.assert_called_once()
        self.assertEqual(mock_cvtcolor.call_count, 2)
        mock_threshold.assert_called_once()

        # Since height < 600, resize should be called
        mock_resize.assert_called_once()

    def test_perform_ocr(self):
        """Test the OCR text extraction function"""
        # Create a mock reader
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = [
            "Line 1_with_underscore",
            " Line 2 with spaces ",
            "",  # Empty line that should be skipped
            "Line 3"
        ]

        # Create dummy image
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Call function
        result = job_card_extractor.perform_ocr(mock_reader, image)

        # Check result
        expected = "Line 1 with underscore\nLine 2 with spaces\nLine 3"
        self.assertEqual(result, expected)

        # Verify reader was called correctly
        mock_reader.readtext.assert_called_once_with(image, detail=0, paragraph=True)

    @patch('cv2.rectangle')
    @patch('cv2.putText')
    def test_create_debug_image(self, mock_puttext, mock_rectangle):
        """Test the debug image creation function"""
        # Create test data
        img_cv = np.zeros((500, 400, 3), dtype=np.uint8)
        lines_y = [0, 100, 200, 500]  # 3 areas
        barcode_annots = [
            ((10, 50, 80, 30), "J12345"),  # x, y, w, h, value
            ((10, 150, 80, 30), "J67890")
        ]
        ocr_annots = [
            (10, "Area 1 Text"),
            (110, "Area 2 Text")
        ]

        # Call function
        result = job_card_extractor.create_debug_image(img_cv, lines_y, barcode_annots, ocr_annots)

        # Verify result is an image with the correct shape
        self.assertEqual(result.shape, img_cv.shape)

        # Verify drawing calls
        # We expect 3 areas to be drawn (2 with enough height, 1 might be skipped due to height)
        # plus 2 barcodes = 5 rectangle calls
        self.assertEqual(mock_rectangle.call_count, 5)

        # 2 barcode annotations + 2 area text annotations
        self.assertEqual(mock_puttext.call_count, 4)

    @patch('job_card_extractor.detect_horizontal_lines')
    @patch('job_card_extractor.detect_barcodes')
    @patch('job_card_extractor.preprocess_image_for_ocr')
    @patch('job_card_extractor.perform_ocr')
    @patch('job_card_extractor.create_debug_image')
    @patch('cv2.cvtColor')
    @patch('cv2.adaptiveThreshold')
    @patch('numpy.array')
    @patch('job_card_extractor.easyocr.Reader')
    def test_process_page(self, mock_easyocr_reader_class, mock_np_array, mock_threshold,
                        mock_cvtcolor, mock_debug_img, mock_perform_ocr, mock_preprocess,
                        mock_detect_barcodes, mock_detect_lines):
        """Test the page processing function"""
        # Set up mocks
        mock_img = MagicMock()
        mock_reader = MagicMock()

        # Configure mocks
        mock_np_array.return_value = np.zeros((300, 400, 3), dtype=np.uint8)

        # Mock adaptiveThreshold
        mock_threshold.return_value = np.zeros((300, 400), dtype=np.uint8)

        # Setup cvtColor to return appropriate values for color conversions
        def cvt_color_side_effect(image, conversion_code):
            if conversion_code == cv2.COLOR_BGR2GRAY:
                return np.zeros((300, 400), dtype=np.uint8)  # Return grayscale
            else:
                return np.zeros((300, 400, 3), dtype=np.uint8)  # Return RGB

        mock_cvtcolor.side_effect = cvt_color_side_effect

        mock_detect_lines.return_value = [0, 100, 300]  # 2 areas

        # Mock barcode detection
        mock_barcode = {
            'type': 'CODE39',
            'barcode': 'J12345',
            'rect': [10, 20, 100, 30]
        }

        # Create proper mock barcode object with rect property
        mock_barcode_object = MagicMock()
        mock_barcode_object.rect = (10, 20, 100, 30)  # Tuple with x, y, w, h
        mock_barcode_object.data = b"J12345"
        mock_barcode_object.type = "CODE39"

        mock_detect_barcodes.side_effect = [
            ([mock_barcode], [mock_barcode_object]),  # First area
            ([], [])                          # Second area
        ]

        # Mock OCR text extraction
        mock_preprocess.return_value = np.zeros((300, 400, 3), dtype=np.uint8)
        mock_perform_ocr.side_effect = [
            "Area 1 Text",   # First area
            "Area 2 Text"    # Second area
        ]

        # Mock debug image creation
        mock_debug_img.return_value = np.zeros((300, 400, 3), dtype=np.uint8)

        # Mock EasyOCR for preview
        mock_easyocr_instance = MagicMock()
        mock_easyocr_instance.readtext.return_value = ["OCR preview text"]
        mock_easyocr_reader_class.return_value = mock_easyocr_instance

        # Call function
        result_areas, result_debug_img = job_card_extractor.process_page(0, mock_img, mock_reader)

        # Verify results
        self.assertEqual(len(result_areas), 2)  # Should have 2 areas
        self.assertEqual(result_areas[0]['page'], 1)  # Page numbers are 1-indexed
        self.assertEqual(result_areas[0]['area_index'], 0)
        self.assertEqual(result_areas[0]['bbox'], [0, 100])
        self.assertEqual(result_areas[0]['ocr_text'], "Area 1 Text")
        self.assertEqual(len(result_areas[0]['barcodes']), 1)
        self.assertEqual(result_areas[0]['barcodes'][0]['barcode'], "J12345")

        # Verify the second area has no barcodes
        self.assertEqual(len(result_areas[1]['barcodes']), 0)

if __name__ == '__main__':
    unittest.main()