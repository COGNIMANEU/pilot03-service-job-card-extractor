#!/usr/bin/env python3
"""Issue #19: EasyOCR must not quantize on x86 CPUs without AVX2.

torch's dynamic int8 LSTM (EasyOCR's default ``quantize=True`` on CPU) dies
with SIGILL on AVX-only x86 CPUs such as the Pilot 03 Xeon E5-1620. The
extractor uses the float recognizer there, keeps EasyOCR's default elsewhere
(Apple Silicon reports the same ``DEFAULT`` capability but quantizes fine),
and honours ``JCE_OCR_QUANTIZE=0|1``.
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import job_card_extractor  # noqa: E402


def _decide(machine, capability, env=None):
    """Run the decision with a mocked machine, torch capability and env."""
    env = {} if env is None else env
    with patch.dict(os.environ, env, clear=False), \
            patch('job_card_extractor.platform.machine', return_value=machine), \
            patch('torch.backends.cpu.get_cpu_capability', return_value=capability):
        if 'JCE_OCR_QUANTIZE' not in env:
            os.environ.pop('JCE_OCR_QUANTIZE', None)
        return job_card_extractor._ocr_quantize_enabled()


class TestOcrQuantizeDecision(unittest.TestCase):
    def test_x86_without_avx2_disables_quantization(self):
        self.assertFalse(_decide('x86_64', 'DEFAULT'))

    def test_windows_amd64_without_avx2_disables_quantization(self):
        self.assertFalse(_decide('AMD64', 'DEFAULT'))

    def test_x86_with_avx2_keeps_quantization(self):
        self.assertTrue(_decide('x86_64', 'AVX2'))
        self.assertTrue(_decide('x86_64', 'AVX512'))

    def test_apple_silicon_keeps_quantization(self):
        # arm64 reports DEFAULT too, but its quantized engine has no AVX2 need.
        self.assertTrue(_decide('arm64', 'DEFAULT'))
        self.assertTrue(_decide('aarch64', 'DEFAULT'))

    def test_override_off_wins_on_capable_cpu(self):
        self.assertFalse(_decide('x86_64', 'AVX2', {'JCE_OCR_QUANTIZE': '0'}))

    def test_override_on_wins_on_avx_only_cpu(self):
        self.assertTrue(_decide('x86_64', 'DEFAULT', {'JCE_OCR_QUANTIZE': '1'}))

    def test_unrecognised_override_falls_back_to_detection(self):
        self.assertFalse(_decide('x86_64', 'DEFAULT', {'JCE_OCR_QUANTIZE': 'maybe'}))

    def test_capability_probe_failure_keeps_easyocr_default(self):
        with patch('job_card_extractor.platform.machine', return_value='x86_64'), \
                patch('torch.backends.cpu.get_cpu_capability', side_effect=RuntimeError('no probe')), \
                patch.dict(os.environ, {}, clear=False):
            os.environ.pop('JCE_OCR_QUANTIZE', None)
            self.assertTrue(job_card_extractor._ocr_quantize_enabled())


class TestReaderReceivesDecision(unittest.TestCase):
    @patch('os.path.exists', return_value=True)
    @patch('job_card_extractor.convert_from_path')
    @patch('job_card_extractor.easyocr.Reader')
    @patch('job_card_extractor.process_page')
    def _run(self, quantize, mock_process_page, mock_reader, mock_convert, _exists):
        mock_convert.return_value = [MagicMock()]
        mock_reader.return_value = MagicMock()
        mock_process_page.return_value = ([{'page': 1, 'area_index': 0}], None)
        with patch('job_card_extractor._ocr_quantize_enabled', return_value=quantize):
            job_card_extractor.extract_areas_from_pdf('doc.pdf', parallel_processing=False)
        return mock_reader

    def test_reader_built_without_quantization_when_disabled(self):
        mock_reader = self._run(False)
        mock_reader.assert_called_once_with(['en'], quantize=False)

    def test_reader_built_with_quantization_when_enabled(self):
        mock_reader = self._run(True)
        mock_reader.assert_called_once_with(['en'], quantize=True)


if __name__ == '__main__':
    unittest.main()
