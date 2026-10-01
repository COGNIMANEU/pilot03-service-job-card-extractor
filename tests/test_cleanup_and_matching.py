#!/usr/bin/env python3
"""Coverage for parent-repo issues #234 (JCE cleanup) and #245 (JCE
decomposition and matching correctness).

- F-BUG-044: a barcode containing "105" must not be assigned to operation "10"
  (decoded-number equality instead of substring containment).
- F-BUG-045: the OCR result cache is safe under parallel page processing.
- F-BUG-046: debug images are named after their real page number.
- F-CLEAN-024/025: no function in job_card_extractor exceeds 80 lines.
- F-CLEAN-026: --no-raw/--no-parallel are wired flags, not no-ops.
"""
import inspect
import os
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so we can import job_card_extractor
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import job_card_extractor


def _area(area_index, ocr_text='', barcodes=None, page=1):
    return {
        'page': page,
        'area_index': area_index,
        'ocr_text': ocr_text,
        'barcodes': [{'barcode': b} for b in (barcodes or [])],
    }


class TestBarcodeMatchingCorrectness(unittest.TestCase):
    """F-BUG-044: matching is by decoded operation number, not substring."""

    def test_barcode_105_not_assigned_to_operation_10(self):
        """A barcode decoding to op 105 must not attach to operation 10."""
        test_data = [
            _area(0, 'Operation 10 CUTTING', barcodes=['J12345Q105']),
            _area(1, 'Operation 105 PACKING', barcodes=[]),
        ]
        result = job_card_extractor.extract_operations(test_data)
        by_number = {op['op_number']: op for op in result}

        self.assertNotEqual(by_number['10']['op_id'], 'J12345Q105')
        self.assertEqual(by_number['105']['op_id'], 'J12345Q105')

    def test_trailing_digit_barcode_not_substring_matched(self):
        """A bare "105" barcode still must not attach to operation 10."""
        test_data = [
            _area(0, 'Operation 10 CUTTING', barcodes=['105']),
        ]
        result = job_card_extractor.extract_operations(test_data)
        self.assertNotEqual(result[0]['op_id'], '105')

    def test_same_area_decoded_match_still_works(self):
        """Regression: a barcode decoding to the op number still matches."""
        test_data = [
            _area(0, 'Operation 10 CUTTING', barcodes=['J12345Q10']),
            _area(1, 'Operation 20 ASSEMBLY', barcodes=['J12345Q20']),
        ]
        result = job_card_extractor.extract_operations(test_data)
        by_number = {op['op_number']: op for op in result}
        self.assertEqual(by_number['10']['op_id'], 'J12345Q10')
        self.assertEqual(by_number['20']['op_id'], 'J12345Q20')
        self.assertEqual(by_number['10']['extraction_strategy'], 'direct_match')

    def test_proximity_match_uses_decoded_number(self):
        """Proximity matching also uses decoded-number equality."""
        test_data = [
            _area(0, 'Operation 30 WELDING', barcodes=[]),
            _area(1, 'Operation 40 MILLING', barcodes=['J12345Q30']),
        ]
        result = job_card_extractor.extract_operations(test_data)
        by_number = {op['op_number']: op for op in result}
        self.assertEqual(by_number['30']['op_id'], 'J12345Q30')
        self.assertEqual(by_number['40']['op_id'], '')

    def test_unrelated_barcode_in_same_area_is_not_claimed(self):
        """The same-area fallback must not claim a barcode owned by another op."""
        test_data = [
            _area(0, 'Operation 10 CUTTING', barcodes=['J12345Q20']),
        ]
        result = job_card_extractor.extract_operations(test_data)
        self.assertEqual(result[0]['op_id'], '')


class TestFunctionLengthLimit(unittest.TestCase):
    """F-CLEAN-024/025: no function exceeds 80 lines of source."""

    def test_no_function_exceeds_80_lines(self):
        def iter_functions():
            for name, obj in inspect.getmembers(job_card_extractor, inspect.isfunction):
                if obj.__module__ == job_card_extractor.__name__:
                    yield name, obj
            for name, obj in inspect.getmembers(
                job_card_extractor.ExtractionLogger, inspect.isfunction
            ):
                yield f"ExtractionLogger.{name}", obj

        offenders = []
        for name, fn in iter_functions():
            length = len(inspect.getsourcelines(fn)[0])
            if length > 80:
                offenders.append(f"{name}: {length} lines")
        self.assertEqual(offenders, [])


class TestCliFlagWiring(unittest.TestCase):
    """F-CLEAN-026: --no-raw/--no-parallel are functional switches."""

    def _run_main(self, argv):
        with patch('job_card_extractor.process_pdf_document') as mock_process:
            mock_process.return_value = {'job_number': 'J1', 'operations': []}
            with patch('sys.argv', ['job_card_extractor.py', *argv, 'doc.pdf']):
                self.assertEqual(job_card_extractor.main(), 0)
        return mock_process.call_args.kwargs

    def test_no_raw_overrides_raw(self):
        """--no-raw changes behavior: it forces raw output back off."""
        with_raw = self._run_main(['--raw'])
        with_both = self._run_main(['--raw', '--no-raw'])
        self.assertTrue(with_raw['save_raw'])
        self.assertFalse(with_both['save_raw'])

    def test_no_parallel_overrides_parallel(self):
        """--no-parallel changes behavior: it disables parallel processing."""
        with_parallel = self._run_main(['--parallel'])
        with_both = self._run_main(['--parallel', '--no-parallel'])
        self.assertTrue(with_parallel['parallel_processing'])
        self.assertFalse(with_both['parallel_processing'])

    def test_defaults_unchanged(self):
        """Defaults stay off: raw output and parallel processing are opt-in."""
        kwargs = self._run_main([])
        self.assertFalse(kwargs['save_raw'])
        self.assertFalse(kwargs['parallel_processing'])
        self.assertTrue(kwargs['save_annotated'])
        self.assertTrue(kwargs['enhance_quality'])


class TestOcrCacheThreadSafety(unittest.TestCase):
    """F-BUG-045: the OCR cache is safe to share across worker threads."""

    def setUp(self):
        job_card_extractor._ocr_cache.clear()

    def tearDown(self):
        job_card_extractor._ocr_cache.clear()

    def _reader(self):
        reader = MagicMock()
        reader.readtext.return_value = [
            ([(0, 0), (10, 0), (10, 10), (0, 10)], 'TEXT', 0.9),
        ]
        return reader

    def test_concurrent_perform_ocr_consistent_results(self):
        import numpy as np
        reader = self._reader()
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        errors = []
        results = []

        def worker():
            try:
                for _ in range(50):
                    results.append(job_card_extractor.perform_ocr(reader, image))
            except Exception as e:  # pragma: no cover - failure path
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        self.assertEqual(set(results), {'TEXT'})
        # The cache must hold at most one entry for this image+reader pair.
        self.assertLessEqual(len(job_card_extractor._ocr_cache), 1)

    def test_cache_hit_avoids_repeat_ocr(self):
        import numpy as np
        reader = self._reader()
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        first = job_card_extractor.perform_ocr(reader, image)
        second = job_card_extractor.perform_ocr(reader, image)
        self.assertEqual(first, second)
        reader.readtext.assert_called_once()


class TestDebugImageNaming(unittest.TestCase):
    """F-BUG-046: debug image filenames carry the real page number."""

    @patch('os.path.exists')
    @patch('job_card_extractor.convert_from_path')
    @patch('job_card_extractor.easyocr.Reader')
    @patch('job_card_extractor.process_page')
    @patch('cv2.imwrite')
    @patch('os.makedirs')
    def test_debug_image_named_by_source_page(
        self, mock_makedirs, mock_imwrite, mock_process_page,
        mock_reader, mock_convert, mock_exists
    ):
        mock_exists.return_value = True
        mock_convert.return_value = [MagicMock(), MagicMock(), MagicMock()]
        mock_reader.return_value = MagicMock()

        debug_img = MagicMock()
        mock_process_page.side_effect = [
            ([{'page': 1, 'area_index': 0}], debug_img),
            RuntimeError('page exploded'),
            ([{'page': 3, 'area_index': 0}], debug_img),
        ]

        with self.assertRaisesRegex(RuntimeError, r'page.*2'):
            job_card_extractor.extract_areas_from_pdf(
                'doc.pdf', output_dir='out', parallel_processing=False
            )

        written = [call.args[0] for call in mock_imwrite.call_args_list]
        self.assertIn(os.path.join('out', 'page_1_areas.jpg'), written)
        self.assertIn(os.path.join('out', 'page_3_areas.jpg'), written)
        # The page-3 image must not be mislabeled as page_2.
        self.assertNotIn(os.path.join('out', 'page_2_areas.jpg'), written)


class TestNoDeadHelpers(unittest.TestCase):
    """F-DEAD-019: dead helpers stay removed."""

    def test_cached_ocr_hash_removed(self):
        self.assertFalse(hasattr(job_card_extractor, '_cached_ocr_hash'))

    def test_version_not_duplicated(self):
        """ExtractionLogger metadata uses __version__, not a literal."""
        with tempfile.TemporaryDirectory() as tmp:
            logger = job_card_extractor.ExtractionLogger(tmp, 'J1')
            self.assertEqual(
                logger.metadata['extraction_info']['extractor_version'],
                job_card_extractor.__version__,
            )
            logger.close_all_loggers()


if __name__ == '__main__':
    unittest.main()
