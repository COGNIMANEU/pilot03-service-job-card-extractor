#!/usr/bin/env python3
import unittest
import sys
import os
import io
from unittest.mock import patch

# Add the parent directory to the path so we can import job_card_extractor
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from job_card_extractor import get_version, display_version, __version__

class TestVersionFunctions(unittest.TestCase):
    def test_get_version(self):
        """Test that get_version returns the correct version string"""
        self.assertEqual(get_version(), __version__)
        self.assertIsInstance(get_version(), str)

    def test_display_version(self):
        """Test that display_version prints the correct version information"""
        # Redirect stdout to capture printed output
        captured_output = io.StringIO()
        with patch('sys.stdout', new=captured_output):
            display_version()

        # Get the printed output
        output = captured_output.getvalue()

        # Check that the output contains the version number
        self.assertIn(f"Job Card Extractor v{__version__}", output)
        # Check that the output contains other expected information
        self.assertIn("Montimage", output)
        self.assertIn("documentation", output)

if __name__ == '__main__':
    unittest.main()