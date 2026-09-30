#!/usr/bin/env python3
"""Honest failure semantics for the job-card extractor (MAS issue #177 / F-BUG-043).

The extractor must fail loudly instead of pretending success:
- the CLI exits non-zero when a document fails,
- no ``*_job_and_operations.json`` result file is written after an exception,
- page-level processing failures are surfaced (which pages failed), and
- internal extraction errors propagate instead of producing empty results.
"""
import glob
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so we can import job_card_extractor
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import job_card_extractor

EXTRACTOR_SCRIPT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'job_card_extractor.py')
)


def result_files(output_dir):
    """Return the list of result files the extractor wrote to output_dir."""
    return glob.glob(os.path.join(output_dir, '*_job_and_operations.json'))


class TestCliExitCodes(unittest.TestCase):
    """The CLI process must tell the OS — and the MAS server — when it failed."""

    def test_corrupt_pdf_exits_nonzero_and_writes_no_result_file(self):
        """A corrupt PDF exits non-zero and writes no result file (issue #177)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = os.path.join(temp_dir, 'corrupt.pdf')
            with open(pdf_path, 'wb') as f:
                f.write(b'this is not a valid PDF document at all')

            output_dir = os.path.join(temp_dir, 'out')
            completed = subprocess.run(
                [sys.executable, EXTRACTOR_SCRIPT, pdf_path, '-o', output_dir],
                capture_output=True,
                timeout=300,
            )

            self.assertNotEqual(
                completed.returncode,
                0,
                f"extractor must exit non-zero on failure; "
                f"stdout={completed.stdout!r} stderr={completed.stderr!r}",
            )
            self.assertEqual(result_files(output_dir), [])

    def test_main_returns_zero_on_success_and_nonzero_on_failure(self):
        """main() reports a per-file failure in its exit code."""
        with patch('job_card_extractor.process_pdf_document') as mock_process:
            mock_process.return_value = {'job_number': 'J1', 'operations': []}
            with patch('sys.argv', ['job_card_extractor.py', 'ok.pdf']):
                self.assertEqual(job_card_extractor.main(), 0)

        with patch('job_card_extractor.process_pdf_document') as mock_process:
            mock_process.side_effect = RuntimeError('boom')
            with patch('sys.argv', ['job_card_extractor.py', 'bad.pdf']):
                self.assertNotEqual(job_card_extractor.main(), 0)

    def test_main_processes_all_files_then_reports_failures(self):
        """A failure on one file must not skip the rest of the batch."""
        with patch('job_card_extractor.process_pdf_document') as mock_process:
            mock_process.side_effect = [
                {'job_number': 'J1', 'operations': []},
                RuntimeError('boom'),
                {'job_number': 'J2', 'operations': []},
            ]
            with patch(
                'sys.argv',
                ['job_card_extractor.py', 'a.pdf', 'b.pdf', 'c.pdf'],
            ):
                exit_code = job_card_extractor.main()

            self.assertEqual(mock_process.call_count, 3)
            self.assertNotEqual(exit_code, 0)


class TestProcessPdfDocumentFailures(unittest.TestCase):
    """process_pdf_document propagates errors instead of returning empty results."""

    @patch('os.path.exists')
    @patch('job_card_extractor.extract_areas_from_pdf')
    def test_area_extraction_error_propagates(self, mock_extract_areas, mock_exists):
        mock_exists.return_value = True
        mock_extract_areas.side_effect = RuntimeError('OCR engine exploded')

        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(RuntimeError):
                job_card_extractor.process_pdf_document(
                    'doc.pdf', output_dir=temp_dir
                )
            self.assertEqual(result_files(temp_dir), [])

    @patch('os.path.exists')
    @patch('job_card_extractor.convert_from_path')
    @patch('job_card_extractor.extract_areas_from_pdf')
    @patch('job_card_extractor.extract_job_and_operations')
    def test_job_extraction_error_propagates_and_writes_no_result_file(
        self, mock_extract_job, mock_extract_areas, mock_convert, mock_exists
    ):
        """Regression: an extraction exception must not produce an empty-operations
        result file — the server would display it as a successful empty card."""
        mock_exists.return_value = True
        mock_extract_areas.return_value = ([{'page': 1, 'ocr_text': 'x'}], [])
        mock_extract_job.side_effect = ValueError('corrupt area data')

        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                job_card_extractor.process_pdf_document(
                    'doc.pdf', output_dir=temp_dir
                )
            self.assertEqual(result_files(temp_dir), [])

    @patch('os.path.exists')
    @patch('job_card_extractor.convert_from_path')
    @patch('job_card_extractor.extract_areas_from_pdf')
    @patch('job_card_extractor.extract_job_and_operations')
    def test_result_file_write_error_propagates(
        self, mock_extract_job, mock_extract_areas, mock_convert, mock_exists
    ):
        """A failed result-file write is a failed extraction, not a warning."""
        mock_exists.return_value = True
        mock_extract_areas.return_value = ([{'page': 1, 'ocr_text': 'x'}], [])
        mock_extract_job.return_value = {
            'job_number': 'J1',
            'operations': [{'op_number': '10'}],
        }
        mock_convert.return_value = [MagicMock()]

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('json.dump', side_effect=OSError('disk full')):
                with self.assertRaises(OSError):
                    job_card_extractor.process_pdf_document(
                        'doc.pdf', output_dir=temp_dir
                    )


class TestPartialPageFailures(unittest.TestCase):
    """Page-level failures are surfaced with the page numbers that failed."""

    def _mock_doc(self, mock_exists, mock_convert, mock_reader, page_count=3):
        mock_exists.return_value = True
        mock_convert.return_value = [MagicMock() for _ in range(page_count)]
        mock_reader.return_value = MagicMock()

    @patch('os.path.exists')
    @patch('job_card_extractor.convert_from_path')
    @patch('job_card_extractor.easyocr.Reader')
    @patch('job_card_extractor.process_page')
    def test_sequential_page_failure_raises_with_page_numbers(
        self, mock_process_page, mock_reader, mock_convert, mock_exists
    ):
        self._mock_doc(mock_exists, mock_convert, mock_reader)
        mock_process_page.side_effect = [
            ([{'page': 1, 'area_index': 0}], None),
            RuntimeError('page exploded'),
            ([{'page': 3, 'area_index': 0}], None),
        ]

        with self.assertRaisesRegex(RuntimeError, r'page.*2'):
            job_card_extractor.extract_areas_from_pdf(
                'doc.pdf', parallel_processing=False
            )

    @patch('os.path.exists')
    @patch('job_card_extractor.convert_from_path')
    @patch('job_card_extractor.easyocr.Reader')
    @patch('job_card_extractor.process_page')
    def test_parallel_page_failure_raises_with_page_numbers(
        self, mock_process_page, mock_reader, mock_convert, mock_exists
    ):
        self._mock_doc(mock_exists, mock_convert, mock_reader)

        def explode_page_two(page_num, img, reader, create_debug=True, enhance_quality=True):
            if page_num == 1:
                raise RuntimeError('page exploded')
            return ([{'page': page_num + 1, 'area_index': 0}], None)

        mock_process_page.side_effect = explode_page_two

        with self.assertRaisesRegex(RuntimeError, r'page.*2'):
            job_card_extractor.extract_areas_from_pdf(
                'doc.pdf', parallel_processing=True
            )

    @patch('job_card_extractor.detect_horizontal_lines')
    @patch('job_card_extractor.cv2.cvtColor')
    def test_process_page_propagates_errors(self, mock_cvtcolor, mock_detect_lines):
        """process_page no longer swallows exceptions into an empty area list."""
        import numpy as np
        mock_cvtcolor.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
        mock_detect_lines.side_effect = RuntimeError('cv exploded')
        with self.assertRaises(RuntimeError):
            job_card_extractor.process_page(0, MagicMock(), MagicMock())


class TestExtractionFailurePropagation(unittest.TestCase):
    """Internal extraction errors propagate instead of yielding empty results."""

    def test_extract_operations_propagates_errors(self):
        with patch(
            'job_card_extractor.re.finditer', side_effect=RuntimeError('regex broke')
        ):
            with self.assertRaises(RuntimeError):
                job_card_extractor.extract_operations(
                    [{'page': 1, 'area_index': 0, 'ocr_text': '10 CUTTING'}]
                )


if __name__ == '__main__':
    unittest.main()
