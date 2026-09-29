@AGENTS.md

# Job Card Extractor — Claude environment

## Commands
- Use Python 3.13 (`requires-python` in `pyproject.toml`, enforced by `install.sh`). The MAS server invokes this repository's `venv/bin/python`; do not create `.venv/` for that integration.
- From this repository root, create the environment with `python3.13 -m venv venv` and install the hash-pinned dependencies with `venv/bin/python -m pip install --require-hashes -r requirements.lock`, then the test tools with `venv/bin/python -m pip install --no-deps --require-hashes -r requirements-dev.lock`.
- Regenerate both locks with the `uv pip compile` commands in `docs/development.md`; do not hand-edit them. `requirements.txt` (loose floors) stays for the MAS installer, which installs from it.
- Run all tests with `venv/bin/python -m pytest tests/` (collects the repository's JCE tests).
- Run one test module with `venv/bin/python -m pytest tests/test_version.py`.
- Check the CLI with `venv/bin/python job_card_extractor.py --version`.

## Gotchas
- Barcode reading needs the native `libzbar` (`apt-get install libzbar0`, `brew install zbar`); the test suite imports `pyzbar` and fails without it.
- PDF conversion needs Poppler on the system (`brew install poppler` on macOS; `apt-get install poppler-utils` on Debian/Ubuntu).
- The first real OCR run downloads EasyOCR language models; the unit tests mock OCR and PDF processing.
- `install.sh` and `install.ps1` create the extractor checkout's `venv/`, matching the MAS server; piped installers clone into the current directory if no checkout is present.
