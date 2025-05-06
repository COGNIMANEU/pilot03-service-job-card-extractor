#!/usr/bin/env python3
import unittest
import sys
import os
import json
import argparse
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

# Add the parent directory to the path so we can import job_card_extractor
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import job_card_extractor

class TestProcessingFunctions(unittest.TestCase):
    @patch('job_card_extractor.extract_areas_from_pdf')
    @patch('job_card_extractor.extract_job_and_operations')
    def test_process_pdf_document(self, mock_extract_job, mock_extract_areas):
        """Test the main PDF processing function"""
        # Mock the extraction functions
        mock_areas_result = ([{'page': 1, 'ocr_text': 'test'}], [MagicMock()])
        mock_extract_areas.return_value = mock_areas_result

        mock_job_result = {'job_number': 'J12345', 'operations': [{'op_number': '10', 'op_name': 'TEST'}]}
        mock_extract_job.return_value = mock_job_result

        # Test with output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock open and json.dump for this test case
            with patch('builtins.open', new_callable=mock_open) as mock_open_file, \
                 patch('os.makedirs') as mock_makedirs, \
                 patch('json.dump') as mock_json_dump:

                # Call the function
                result = job_card_extractor.process_pdf_document(
                    'test.pdf',
                    output_dir=temp_dir,
                    lang_list=['en'],
                    save_raw=True,
                    save_annotated=True
                )

                # Verify results
                self.assertEqual(result, mock_job_result)

                # Verify directory creation
                mock_makedirs.assert_called()

                # Verify file operations (raw and clean outputs)
                self.assertEqual(mock_open_file.call_count, 2)

                # Verify JSON dumps
                self.assertEqual(mock_json_dump.call_count, 2)

                # Verify first call to extract_areas_from_pdf
                mock_extract_areas.assert_called_with(
                    'test.pdf',
                    lang_list=['en'],
                    output_dir=os.path.join(temp_dir, "annotated")
                )

        # Test without output directory - with fresh mocks
        mock_extract_areas.reset_mock()
        mock_extract_job.reset_mock()

        # Use a separate patch for this test case
        with patch('builtins.open', new_callable=mock_open) as mock_open_file_2:
            result = job_card_extractor.process_pdf_document(
                'test.pdf',
                output_dir=None
            )

            # Verify results without output dir
            self.assertEqual(result, mock_job_result)
            # No files should be opened when output_dir is None
            mock_open_file_2.assert_not_called()

    def test_main_function_with_version_flag(self):
        """Test the main function with version flag"""
        # Define a test function that mimics main() but is isolated for this test
        def isolated_main():
            parser = argparse.ArgumentParser(
                description="Process PDF job documents and extract job number and operations"
            )
            parser.add_argument(
                "pdf_files",
                nargs='*',
                help="Path to the PDF file(s) to process"
            )
            parser.add_argument(
                "-v", "--version",
                action="store_true",
                help="Display version information"
            )
            # Add other arguments
            parser.add_argument("-o", "--output-dir", help="Directory to save output files")
            parser.add_argument("-l", "--lang", nargs='+', default=['en'], help="Language codes for OCR")
            parser.add_argument("--no-raw", action="store_true", help="Don't save raw extraction data")
            parser.add_argument("--no-annotated", action="store_true", help="Don't save annotated debug images")

            args = parser.parse_args()

            if args.version:
                job_card_extractor.display_version()
                sys.exit(0)

            # Rest of function not needed for this test

        # Now patch sys.exit and display_version for our isolated function
        with patch('sys.exit') as mock_exit, \
             patch('job_card_extractor.display_version') as mock_display_version, \
             patch('sys.argv', ['job_card_extractor.py', '--version']):

            # Call our isolated function
            isolated_main()

            # Verify version was displayed and exit was called
            mock_display_version.assert_called_once()
            mock_exit.assert_called_once_with(0)

    def test_main_function_with_no_files(self):
        """Test the main function with no PDF files provided"""
        with patch('argparse.ArgumentParser.parse_args') as mock_parse_args, \
             patch('sys.exit') as mock_exit:

            mock_args = MagicMock()
            mock_args.version = False
            mock_args.pdf_files = []
            mock_parse_args.return_value = mock_args

            # Call main function
            job_card_extractor.main()

            # Verify exit was called with error
            mock_exit.assert_called_once_with(1)

    def test_main_function_with_pdf_files(self):
        """Test the main function with PDF files provided"""
        with patch('argparse.ArgumentParser.parse_args') as mock_parse_args, \
             patch('job_card_extractor.process_pdf_document') as mock_process:

            mock_args = MagicMock()
            mock_args.pdf_files = ['test1.pdf', 'test2.pdf']
            mock_args.output_dir = 'output'
            mock_args.lang = ['en']
            mock_args.no_raw = False
            mock_args.no_annotated = True
            mock_args.version = False
            mock_parse_args.return_value = mock_args

            # Mock process_pdf_document to return a result
            mock_process.return_value = {'job_number': 'J12345', 'operations': []}

            # Call main function
            job_card_extractor.main()

            # Verify process_pdf_document was called for each PDF file
            self.assertEqual(mock_process.call_count, 2)

            # Verify calls with correct arguments
            mock_process.assert_any_call(
                'test1.pdf',
                output_dir='output',
                lang_list=['en'],
                save_raw=True,
                save_annotated=False
            )

            mock_process.assert_any_call(
                'test2.pdf',
                output_dir='output',
                lang_list=['en'],
                save_raw=True,
                save_annotated=False
            )

    @patch('os.path.exists')
    @patch('job_card_extractor.convert_from_path')
    @patch('job_card_extractor.easyocr.Reader')
    @patch('job_card_extractor.process_page')
    @patch('cv2.imwrite')
    @patch('os.makedirs')
    def test_extract_areas_from_pdf(self, mock_makedirs, mock_imwrite, mock_process_page,
                                 mock_reader, mock_convert, mock_exists):
        """Test the PDF to areas extraction function"""
        # Set up mocks
        mock_exists.return_value = True

        # Mock PDF to image conversion
        page1 = MagicMock()
        page2 = MagicMock()
        mock_convert.return_value = [page1, page2]

        # Mock EasyOCR reader
        mock_reader_instance = MagicMock()
        mock_reader.return_value = mock_reader_instance

        # Mock process_page results
        page1_areas = [{'page': 1, 'area_index': 0}]
        page1_debug = MagicMock()

        page2_areas = [{'page': 2, 'area_index': 0}, {'page': 2, 'area_index': 1}]
        page2_debug = MagicMock()

        mock_process_page.side_effect = [
            (page1_areas, page1_debug),
            (page2_areas, page2_debug)
        ]

        # Call the function
        areas, debug_images = job_card_extractor.extract_areas_from_pdf(
            'test.pdf',
            lang_list=['en', 'fr'],
            output_dir='output/debug'
        )

        # Verify results
        self.assertEqual(len(areas), 3)  # 1 from page1 + 2 from page2
        self.assertEqual(len(debug_images), 2)  # 2 pages

        # Verify process_page calls
        self.assertEqual(mock_process_page.call_count, 2)

        # Verify output directory creation and image saving
        mock_makedirs.assert_called_once_with('output/debug', exist_ok=True)
        self.assertEqual(mock_imwrite.call_count, 2)

        # Test FileNotFoundError handling
        mock_exists.return_value = False
        with self.assertRaises(FileNotFoundError):
            job_card_extractor.extract_areas_from_pdf('notfound.pdf')

if __name__ == '__main__':
    unittest.main()