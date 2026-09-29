"""Keep the dependency manifests consistent without installing anything."""

import re
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def requirement_lines(path):
    lines = (ROOT / path).read_text(encoding='utf-8').splitlines()
    return [line.strip() for line in lines if line.strip() and not line.lstrip().startswith('#')]


class TestDependencyManifests(unittest.TestCase):
    def test_pyproject_declares_python_313_and_matches_requirements_txt(self):
        project = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))['project']
        self.assertEqual(project['requires-python'], '>=3.13,<3.14')
        self.assertEqual(sorted(project['dependencies']), sorted(requirement_lines('requirements.txt')))

    def test_obsolete_argparse_backport_is_not_declared(self):
        # argparse ships with the standard library; the PyPI backport is obsolete.
        for name in ('requirements.txt', 'requirements.lock', 'requirements-dev.lock'):
            for line in requirement_lines(name):
                self.assertIsNone(re.match(r'argparse\b', line), f'{name}: {line}')

    def test_lockfiles_pin_every_package_with_hashes(self):
        for name in ('requirements.lock', 'requirements-dev.lock'):
            text = (ROOT / name).read_text(encoding='utf-8')
            packages = re.findall(r'^([A-Za-z0-9._-]+)==', text, flags=re.MULTILINE)
            self.assertGreater(len(packages), 5, name)
            # Every pinned package is followed by at least one hash before the next.
            blocks = re.split(r'^(?=[A-Za-z0-9._-]+==)', text, flags=re.MULTILINE)[1:]
            self.assertEqual(len(blocks), len(packages), name)
            for block in blocks:
                self.assertIn('--hash=sha256:', block, f'{name}: {block.splitlines()[0]}')
        self.assertIn('pytest', (ROOT / 'requirements-dev.lock').read_text(encoding='utf-8'))
