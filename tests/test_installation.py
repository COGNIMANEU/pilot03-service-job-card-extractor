"""Installer/documentation regression tests without PDF/OCR dependencies."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TestStandaloneInstaller(unittest.TestCase):
    """Keep installer paths aligned with the MAS server's checkout-local venv."""

    def test_readme_and_windows_installer_use_checkout_venv(self):
        readme = (ROOT / 'README.md').read_text(encoding='utf-8')
        windows = (ROOT / 'install.ps1').read_text(encoding='utf-8')
        self.assertIn('python -m venv venv', readme)
        self.assertIn('source venv/bin/activate', readme)
        self.assertIn('$venvPath = Join-Path $checkout "venv"', windows)
        self.assertNotIn('$env:USERPROFILE\\.venv', windows)
        self.assertNotIn('python -m venv .venv', readme)

    def test_shell_installer_uses_checkout_for_dependencies(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            checkout = base / 'pilot03-service-job-card-extractor'
            checkout.mkdir()
            (checkout / 'job_card_extractor.py').touch()
            (checkout / 'requirements.txt').write_text('example-dependency\n', encoding='utf-8')
            python = checkout / 'venv' / 'bin' / 'python'
            python.parent.mkdir(parents=True)
            python.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$PIP_LOG"\n', encoding='utf-8')
            python.chmod(0o755)
            # Source a downloaded copy from outside the checkout so the pipe
            # fallback resolves the caller's existing checkout, not the script.
            script = base / 'downloaded-installer.sh'
            script.write_bytes((ROOT / 'install.sh').read_bytes())
            log = base / 'pip-args'
            env = dict(os.environ, PIP_LOG=str(log), HOME=str(base / 'home'))
            result = subprocess.run(
                ['bash', '-c', 'source "$1"; checkout="$(resolve_checkout)"; '
                 'install_job_card_extractor "$checkout"', 'bash', str(script)],
                cwd=base, env=env, capture_output=True, text=True, check=True,
            )
            self.assertIn(f'source {checkout}/venv/bin/activate', result.stdout)
            self.assertIn(f'-m pip install -r {checkout}/requirements.txt',
                          log.read_text(encoding='utf-8'))
            self.assertFalse((base / 'home' / '.venv').exists())

    def test_piped_shell_installer_runs_and_rejects_incomplete_checkout(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            (base / 'pilot03-service-job-card-extractor').mkdir()
            result = subprocess.run(
                ['bash'], input=(ROOT / 'install.sh').read_text(encoding='utf-8'),
                cwd=base, capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Extractor checkout incomplete', result.stderr)
            self.assertNotIn('unbound variable', result.stderr)

    def test_shell_installer_fails_on_incomplete_checkout(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            (base / 'pilot03-service-job-card-extractor').mkdir()
            script = base / 'downloaded-installer.sh'
            script.write_bytes((ROOT / 'install.sh').read_bytes())
            result = subprocess.run(
                ['bash', '-c', 'source "$1"; resolve_checkout', 'bash', str(script)],
                cwd=base, capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Extractor checkout incomplete', result.stderr)


if __name__ == '__main__':
    unittest.main()
