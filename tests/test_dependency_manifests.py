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
        for name in ('requirements.lock', 'requirements/cpu.lock', 'requirements-dev.lock'):
            text = (ROOT / name).read_text(encoding='utf-8')
            packages = re.findall(r'^([A-Za-z0-9._-]+)==', text, flags=re.MULTILINE)
            self.assertGreater(len(packages), 5, name)
            # Every pinned package is followed by at least one hash before the next.
            blocks = re.split(r'^(?=[A-Za-z0-9._-]+==)', text, flags=re.MULTILINE)[1:]
            self.assertEqual(len(blocks), len(packages), name)
            for block in blocks:
                self.assertIn('--hash=sha256:', block, f'{name}: {block.splitlines()[0]}')
        self.assertIn('pytest', (ROOT / 'requirements-dev.lock').read_text(encoding='utf-8'))

    def test_cpu_lock_has_no_cuda_packages_and_cpu_torch_on_linux(self):
        text = (ROOT / 'requirements/cpu.lock').read_text(encoding='utf-8')
        packages = re.findall(r'^([A-Za-z0-9._-]+)==', text, flags=re.MULTILINE)
        cuda = [p for p in packages if re.match(r'(nvidia-|cuda-|triton$)', p)]
        self.assertEqual(cuda, [])
        for name in ('torch', 'torchvision'):
            self.assertRegex(text, rf"(?m)^{name}==[0-9.]+\+cpu ; sys_platform != 'darwin'")
        self.assertIn('--extra-index-url https://pypi.org/simple', text)
        self.assertIn('--index-url https://download.pytorch.org/whl/cpu', text)

    def test_cpu_lock_matches_main_lock_versions_and_hashes(self):
        # Built with -c requirements.lock: shared packages keep the same pin
        # and every PyPI file hash (an index mirror must not narrow them).
        def parse(name):
            text = (ROOT / name).read_text(encoding='utf-8')
            blocks = re.split(r'^(?=[A-Za-z0-9._-]+==)', text, flags=re.MULTILINE)[1:]
            pins = {}
            for block in blocks:
                pin = block.split()[0]
                pins[pin] = set(re.findall(r'sha256:(\w+)', block))
            return pins
        main, cpu = parse('requirements.lock'), parse('requirements/cpu.lock')
        for pin, hashes in cpu.items():
            if pin.split('==')[0] in ('torch', 'torchvision'):
                continue
            self.assertIn(pin, main)
            self.assertLessEqual(main[pin], hashes, pin)
