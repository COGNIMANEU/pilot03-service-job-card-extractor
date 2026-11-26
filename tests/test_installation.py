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
            # Resolve symlinks (macOS /var -> /private/var): the installer
            # prints the physical checkout path.
            base = Path(temporary).resolve()
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

    def test_shell_installer_prefers_hash_pinned_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            checkout = base / 'pilot03-service-job-card-extractor'
            checkout.mkdir()
            (checkout / 'job_card_extractor.py').touch()
            (checkout / 'requirements.txt').write_text('example-dependency\n', encoding='utf-8')
            (checkout / 'requirements.lock').write_text('example-dependency==1.0\n', encoding='utf-8')
            python = checkout / 'venv' / 'bin' / 'python'
            python.parent.mkdir(parents=True)
            python.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$PIP_LOG"\n', encoding='utf-8')
            python.chmod(0o755)
            script = base / 'downloaded-installer.sh'
            script.write_bytes((ROOT / 'install.sh').read_bytes())
            log = base / 'pip-args'
            env = dict(os.environ, PIP_LOG=str(log), HOME=str(base / 'home'))
            subprocess.run(
                ['bash', '-c', 'source "$1"; install_job_card_extractor "$2"',
                 'bash', str(script), str(checkout)],
                cwd=base, env=env, capture_output=True, text=True, check=True,
            )
            self.assertIn(f'-m pip install --require-hashes -r {checkout}/requirements.lock',
                          log.read_text(encoding='utf-8'))

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


def fake_nvidia_smi(bin_dir, output, exit_code=0):
    """Write an nvidia-smi stub that answers the installer's GPU query."""
    bin_dir.mkdir(exist_ok=True)
    stub = bin_dir / 'nvidia-smi'
    stub.write_text(
        '#!/bin/sh\n'
        'case "$*" in *--query-gpu=driver_version,compute_cap*) ;; *) exit 2 ;; esac\n'
        f"printf '%s' '{output}'\n"
        f'exit {exit_code}\n',
        encoding='utf-8',
    )
    stub.chmod(0o755)


class TestTorchVariantSelection(unittest.TestCase):
    """install.sh picks requirements/cpu.lock unless the GPU runs CUDA 13 torch."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name).resolve()
        self.checkout = self.base / 'checkout'
        (self.checkout / 'requirements').mkdir(parents=True)
        for name in ('requirements.txt', 'requirements.lock', 'requirements/cpu.lock'):
            (self.checkout / name).write_text('example==1.0\n', encoding='utf-8')
        self.fake_bin = self.base / 'fakebin'

    def run_installer(self, command, os_name='linux', with_fake_bin=True, **env_extra):
        path = '/usr/bin:/bin'
        if with_fake_bin:
            path = f'{self.fake_bin}:{path}'
        env = {'PATH': path, 'HOME': str(self.base), **env_extra}
        return subprocess.run(
            ['bash', '-c', f'source "$1"; {command}', 'bash', str(ROOT / 'install.sh'),
             str(self.checkout), os_name],
            cwd=self.base, env=env, capture_output=True, text=True,
        )

    def select(self, **kwargs):
        return self.run_installer('select_lockfile "$2" "$3"', **kwargs)

    def test_pilot_gpu_driver_535_cc_61_picks_cpu(self):
        fake_nvidia_smi(self.fake_bin, '535.183.01, 6.1\n')
        result = self.select()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), 'requirements/cpu.lock')
        self.assertIn('torch variant: cpu (driver 535 < 580 required by CUDA 13)', result.stderr)

    def test_driver_580_cc_86_picks_cuda(self):
        fake_nvidia_smi(self.fake_bin, '580.65.06, 8.6\n')
        result = self.select()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), 'requirements.lock')
        self.assertIn('torch variant: cuda (driver 580, compute capability 8.6)', result.stderr)

    def test_new_driver_on_pascal_gpu_picks_cpu(self):
        fake_nvidia_smi(self.fake_bin, '580.65.06, 6.1\n')
        result = self.select()
        self.assertEqual(result.stdout.strip(), 'requirements/cpu.lock')
        self.assertIn('compute capability 6.1 < 7.5 required by CUDA 13 torch', result.stderr)

    def test_compute_capability_compares_numerically(self):
        # "10.0" sorts before "7.5" as a string; Blackwell must still pick cuda.
        fake_nvidia_smi(self.fake_bin, '590.10, 10.0\n')
        self.assertEqual(self.select().stdout.strip(), 'requirements.lock')

    def test_any_capable_gpu_picks_cuda(self):
        fake_nvidia_smi(self.fake_bin, '580.65.06, 6.1\n580.65.06, 8.6\n')
        self.assertEqual(self.select().stdout.strip(), 'requirements.lock')

    def test_missing_nvidia_smi_picks_cpu(self):
        result = self.select(with_fake_bin=False)
        self.assertEqual(result.stdout.strip(), 'requirements/cpu.lock')
        self.assertIn('torch variant: cpu (nvidia-smi not found)', result.stderr)

    def test_failing_or_unreadable_nvidia_smi_picks_cpu(self):
        fake_nvidia_smi(self.fake_bin, '', exit_code=9)
        result = self.select()
        self.assertEqual(result.stdout.strip(), 'requirements/cpu.lock')
        self.assertIn('nvidia-smi failed', result.stderr)
        fake_nvidia_smi(self.fake_bin, '[N/A], [N/A]\n')
        result = self.select()
        self.assertEqual(result.stdout.strip(), 'requirements/cpu.lock')
        self.assertIn('unreadable nvidia-smi output', result.stderr)

    def test_macos_picks_cpu_even_with_capable_gpu(self):
        fake_nvidia_smi(self.fake_bin, '580.65.06, 8.6\n')
        result = self.select(os_name='macos')
        self.assertEqual(result.stdout.strip(), 'requirements/cpu.lock')
        self.assertIn('torch variant: cpu (macos', result.stderr)

    def test_lock_file_override_is_used(self):
        fake_nvidia_smi(self.fake_bin, '580.65.06, 8.6\n')
        result = self.select(LOCK_FILE='requirements/cpu.lock')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), 'requirements/cpu.lock')
        self.assertIn('torch variant: LOCK_FILE=requirements/cpu.lock', result.stderr)
        self.assertNotIn('WARN', result.stderr)

    def test_cuda_override_on_cpu_host_installs_cuda_with_warning(self):
        fake_nvidia_smi(self.fake_bin, '535.183.01, 6.1\n')
        result = self.select(LOCK_FILE='requirements.lock')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), 'requirements.lock')
        self.assertIn('WARN', result.stderr)
        self.assertIn('auto picks cpu (driver 535 < 580', result.stderr)

    def test_missing_lock_file_override_fails(self):
        result = self.select(LOCK_FILE='requirements/missing.lock')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('LOCK_FILE=requirements/missing.lock not found', result.stderr)

    def test_falls_back_when_cpu_lock_or_all_locks_missing(self):
        (self.checkout / 'requirements' / 'cpu.lock').unlink()
        result = self.select(with_fake_bin=False)
        self.assertEqual(result.stdout.strip(), 'requirements.lock')
        self.assertIn('requirements/cpu.lock not found', result.stderr)
        (self.checkout / 'requirements.lock').unlink()
        result = self.select(with_fake_bin=False)
        self.assertEqual(result.stdout.strip(), 'requirements.txt')
        self.assertIn('installing unpinned requirements.txt', result.stderr)

    def test_install_uses_selected_lock_with_hashes(self):
        python = self.checkout / 'venv' / 'bin' / 'python'
        python.parent.mkdir(parents=True)
        python.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$PIP_LOG"\n', encoding='utf-8')
        python.chmod(0o755)
        log = self.base / 'pip-args'
        fake_nvidia_smi(self.fake_bin, '535.183.01, 6.1\n')
        result = self.run_installer('install_job_card_extractor "$2" "$3"', PIP_LOG=str(log))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f'-m pip install --require-hashes -r {self.checkout}/requirements/cpu.lock',
                      log.read_text(encoding='utf-8'))


class TestWindowsInstallerLockSelection(unittest.TestCase):
    """install.ps1 cannot run here; check its lock choice textually."""

    def test_windows_installer_defaults_to_cpu_lock_with_override(self):
        windows = (ROOT / 'install.ps1').read_text(encoding='utf-8')
        self.assertIn('$CPU_LOCK = "requirements/cpu.lock"', windows)
        self.assertIn('$env:LOCK_FILE', windows)
        self.assertIn('Write-Info "torch variant: cpu $reason"', windows)
        self.assertIn('& $venvPython -m pip install --require-hashes -r $lockFile', windows)


if __name__ == '__main__':
    unittest.main()
