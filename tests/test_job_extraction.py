#!/usr/bin/env python3
import unittest
import sys
import os
import json
from unittest.mock import patch, MagicMock, mock_open

# Add the parent directory to the path so we can import job_card_extractor
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import job_card_extractor

class TestJobExtractionFunctions(unittest.TestCase):
    def test_extract_job_number(self):
        """Test the job number extraction function with various inputs"""
        # Test case 1: Job number in area with 'Job No' text
        test_data_1 = [
            {
                'page': 1,
                'area_index': 0,
                'ocr_text': 'Job No: 12345',
                'barcodes': [{'barcode': 'J12345'}, {'barcode': 'J67890'}]
            },
            {
                'page': 1,
                'area_index': 1,
                'ocr_text': 'Other info',
                'barcodes': [{'barcode': 'OTHER1'}]
            }
        ]
        self.assertEqual(job_card_extractor.extract_job_number(test_data_1), 'J12345')

        # Test case 2: No 'Job No' text but barcode on first page
        test_data_2 = [
            {
                'page': 1,
                'area_index': 0,
                'ocr_text': 'Some text',
                'barcodes': [{'barcode': 'J12345'}]
            }
        ]
        self.assertEqual(job_card_extractor.extract_job_number(test_data_2), 'J12345')

        # Test case 3: No barcodes on first page
        test_data_3 = [
            {
                'page': 1,
                'area_index': 0,
                'ocr_text': 'Job No: 12345',
                'barcodes': []
            },
            {
                'page': 2,
                'area_index': 0,
                'ocr_text': 'Some text',
                'barcodes': [{'barcode': 'J12345'}]
            }
        ]
        self.assertEqual(job_card_extractor.extract_job_number(test_data_3), '')

        # Test case 4: Empty data
        self.assertEqual(job_card_extractor.extract_job_number([]), '')

    def test_clean_operation_name(self):
        """Test the operation name cleaning function"""
        # Test removing year prefix
        self.assertEqual(job_card_extractor.clean_operation_name('2022 Operation Name'), 'Operation Name')

        # Test removing scan instructions (standard format)
        self.assertEqual(job_card_extractor.clean_operation_name('Operation Name Scan barcodes to start job operation'),
                        'Operation Name')

        # Test removing scan instructions (hyphenated format)
        self.assertEqual(job_card_extractor.clean_operation_name('Operation Name ~Scan-barcodes-to-start-job operation'),
                        'Operation Name')

        # Test removing general scan suffix
        self.assertEqual(job_card_extractor.clean_operation_name('Operation Name Scan'),
                        'Operation Name')

    def test_extract_operations(self):
        """Test the operations extraction function"""
        # Test data with different operation formats
        test_data = [
            # Area with "Operation XX Name" format
            {
                'page': 1,
                'area_index': 0,
                'ocr_text': 'Operation 10 CUTTING',
                'barcodes': [{'barcode': 'J12345Q10'}]
            },
            # Area with "XX Name" format
            {
                'page': 1,
                'area_index': 1,
                'ocr_text': '20 ASSEMBLY\nScan barcodes to start job operation',
                'barcodes': [{'barcode': 'J12345Q20'}]
            },
            # Area with operation number on one line and name on next line
            {
                'page': 1,
                'area_index': 2,
                'ocr_text': '30\nWELDING',
                'barcodes': []
            },
            # Area with non-operation text
            {
                'page': 1,
                'area_index': 3,
                'ocr_text': 'This is not an operation',
                'barcodes': []
            },
            # Area with too large operation number (should be ignored)
            {
                'page': 1,
                'area_index': 4,
                'ocr_text': '2022 YEAR INFO',
                'barcodes': []
            }
        ]

        # Call the function
        result = job_card_extractor.extract_operations(test_data)

        # Verify results
        self.assertEqual(len(result), 3)  # Should have 3 operations (10, 20, 30)

        # Check operation 10
        self.assertEqual(result[0]['op_number'], '10')
        self.assertEqual(result[0]['op_name'], 'CUTTING')
        self.assertEqual(result[0]['op_id'], 'J12345Q10')

        # Check operation 20
        self.assertEqual(result[1]['op_number'], '20')
        self.assertEqual(result[1]['op_name'], 'ASSEMBLY')
        self.assertEqual(result[1]['op_id'], 'J12345Q20')

        # Check operation 30
        self.assertEqual(result[2]['op_number'], '30')
        self.assertEqual(result[2]['op_name'], 'WELDING')
        self.assertEqual(result[2]['op_id'], '')  # No barcode for this operation

    def test_extract_job_details(self):
        """Test the job details extraction function with various inputs"""
        # Test case 1: Job details with all information present
        test_data_1 = [
            {
                'page': 1,
                'area_index': 0,
                'ocr_text': 'Job No: 12345',
                'barcodes': [{'barcode': 'J12345'}, {'barcode': 'J67890'}]
            },
            {
                'page': 1,
                'area_index': 1,
                'ocr_text': 'Quantity: 500',
                'barcodes': []
            },
            {
                'page': 1,
                'area_index': 2,
                'ocr_text': 'Delivery Date: 15/06/2025',
                'barcodes': []
            },
            {
                'page': 1,
                'area_index': 3,
                'ocr_text': 'Operation 10 CUTTING',
                'barcodes': [{'barcode': 'J12345Q10'}]
            }
        ]
        result_1 = job_card_extractor.extract_job_details(test_data_1)
        self.assertEqual(result_1['job_number'], 'J12345')
        self.assertEqual(result_1['quantity'], '500')
        self.assertEqual(result_1['delivery_date'], '15/06/2025')

        # Test case 2: Job details with alternate format
        test_data_2 = [
            {
                'page': 1,
                'area_index': 0,
                'ocr_text': 'Some text',
                'barcodes': [{'barcode': 'J54321'}]
            },
            {
                'page': 1,
                'area_index': 1,
                'ocr_text': 'QTY: 250.00',
                'barcodes': []
            },
            {
                'page': 1,
                'area_index': 2,
                'ocr_text': 'Date Required: 10-May-2025',
                'barcodes': []
            }
        ]
        result_2 = job_card_extractor.extract_job_details(test_data_2)
        self.assertEqual(result_2['job_number'], 'J54321')
        self.assertEqual(result_2['quantity'], '250.00')
        self.assertEqual(result_2['delivery_date'], '10-May-2025')

        # Test case 3: Missing quantity and delivery date
        test_data_3 = [
            {
                'page': 1,
                'area_index': 0,
                'ocr_text': 'Job No: 12345',
                'barcodes': [{'barcode': 'J98765'}]
            }
        ]
        result_3 = job_card_extractor.extract_job_details(test_data_3)
        self.assertEqual(result_3['job_number'], 'J98765')
        self.assertEqual(result_3['quantity'], '')
        self.assertEqual(result_3['delivery_date'], '')

        # Test case 4: Empty data
        result_4 = job_card_extractor.extract_job_details([])
        self.assertEqual(result_4['job_number'], '')
        self.assertEqual(result_4['quantity'], '')
        self.assertEqual(result_4['delivery_date'], '')

    def test_extract_job_and_operations(self):
        """Test the combined job and operations extraction function"""
        # Simple test data
        test_data = [
            {
                'page': 1,
                'area_index': 0,
                'ocr_text': 'Job No: J12345',
                'barcodes': [{'barcode': 'J12345'}]
            },
            {
                'page': 1,
                'area_index': 1,
                'ocr_text': 'Quantity: 100\nDelivery Date: 30/06/2025',
                'barcodes': []
            },
            {
                'page': 1,
                'area_index': 2,
                'ocr_text': 'Operation 10 CUTTING',
                'barcodes': [{'barcode': 'J12345Q10'}]
            }
        ]

        # Call the function
        result = job_card_extractor.extract_job_and_operations(test_data)

        # Verify results
        self.assertIsInstance(result, dict)
        self.assertIn('job_number', result)
        self.assertIn('quantity', result)
        self.assertIn('delivery_date', result)
        self.assertIn('operations', result)

        self.assertEqual(result['job_number'], 'J12345')
        self.assertEqual(result['quantity'], '100')
        self.assertEqual(result['delivery_date'], '30/06/2025')
        self.assertEqual(len(result['operations']), 1)
        self.assertEqual(result['operations'][0]['op_number'], '10')
        self.assertEqual(result['operations'][0]['op_name'], 'CUTTING')

        # Test with empty data
        empty_result = job_card_extractor.extract_job_and_operations([])
        self.assertEqual(empty_result['job_number'], '')
        self.assertEqual(empty_result['quantity'], '')
        self.assertEqual(empty_result['delivery_date'], '')
        self.assertEqual(len(empty_result['operations']), 0)

if __name__ == '__main__':
    unittest.main()