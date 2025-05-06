#!/usr/bin/env python3
import unittest
import sys
import os
import numpy as np
from unittest.mock import patch, MagicMock
from PIL import Image

# Add the parent directory to the path so we can import job_card_extractor
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import job_card_extractor

class TestBarcodeExtractionFunctions(unittest.TestCase):
    def test_clean_barcode_value(self):
        """Test the clean_barcode_value function with various inputs"""
        # Test with a simple alphanumeric string
        self.assertEqual(job_card_extractor.clean_barcode_value("J12345"), "J12345")

        # Test with control characters and spaces
        self.assertEqual(job_card_extractor.clean_barcode_value("J123 45\t\n"), "J12345")

        # Test with special characters
        self.assertEqual(job_card_extractor.clean_barcode_value("J-123.45#"), "J12345")

        # Test with empty string
        self.assertEqual(job_card_extractor.clean_barcode_value(""), "")

    @patch('job_card_extractor.decode')
    def test_detect_barcodes(self, mock_decode):
        """Test the detect_barcodes function with mocked barcode detection"""
        # Create a mock barcode result
        mock_barcode = MagicMock()
        mock_barcode.type = "CODE39"
        mock_barcode.data = b"J123456"
        mock_barcode.rect = (10, 20, 100, 30)  # x, y, width, height

        # Configure the mock to return our mock barcode
        mock_decode.return_value = [mock_barcode]

        # Create a dummy image
        img_crop = np.zeros((100, 200, 3), dtype=np.uint8)

        # Call the function
        result, raw_barcodes = job_card_extractor.detect_barcodes(img_crop)

        # Verify results
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['type'], "CODE39")
        self.assertEqual(result[0]['barcode'], "J123456")
        self.assertEqual(result[0]['rect'], [10, 20, 100, 30])

        # Verify decode was called with the correct argument type
        args, kwargs = mock_decode.call_args
        self.assertIsInstance(args[0], Image.Image)

    @patch('cv2.cvtColor')
    @patch('cv2.findContours')
    @patch('cv2.adaptiveThreshold')
    @patch('cv2.morphologyEx')
    @patch('cv2.getStructuringElement')
    def test_detect_horizontal_lines(self, mock_get_struct, mock_morphology,
                                   mock_threshold, mock_contours, mock_cvt_color):
        """Test detect_horizontal_lines function with mocked OpenCV calls"""
        # Set up mocks
        mock_img = MagicMock()
        mock_img.shape = (800, 600, 3)  # height, width, channels

        mock_cvt_color.return_value = np.zeros((800, 600), dtype=np.uint8)
        mock_threshold.return_value = np.zeros((800, 600), dtype=np.uint8)
        mock_morphology.return_value = np.zeros((800, 600), dtype=np.uint8)

        # Mock contours to return sample lines at y=100, y=250, y=400
        mock_cnt1 = MagicMock()
        mock_cnt2 = MagicMock()
        mock_cnt3 = MagicMock()
        mock_cnt1.__getitem__.return_value = np.array([[[0, 100], [600, 100]]])
        mock_cnt2.__getitem__.return_value = np.array([[[0, 250], [600, 250]]])
        mock_cnt3.__getitem__.return_value = np.array([[[0, 400], [600, 400]]])

        # Set up cv2.boundingRect returns for each contour
        # Format: (x, y, width, height)
        mock_contours.return_value = ([mock_cnt1, mock_cnt2, mock_cnt3], None)

        with patch('cv2.boundingRect', side_effect=[
            (0, 100, 600, 2),  # First contour rectangle
            (0, 250, 600, 2),  # Second contour rectangle
            (0, 400, 600, 2),  # Third contour rectangle
        ]):
            # Call the function
            result = job_card_extractor.detect_horizontal_lines(mock_img)

            # Verify the result includes the starting position, detected lines, and end position
            self.assertEqual(result, [0, 100, 250, 400, 800])

            # Verify the morphological kernel was created with expected dimensions
            mock_get_struct.assert_called_once()
            args, kwargs = mock_get_struct.call_args
            # First arg should be MORPH_RECT
            # Second and third args should be width and height of kernel
            self.assertEqual(args[1], (120, 2))  # width = img_width / 5 = 600 / 5 = 120

if __name__ == '__main__':
    unittest.main()