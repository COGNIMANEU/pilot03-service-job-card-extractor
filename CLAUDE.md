@AGENTS.md

# Job Card Extractor — Claude environment

## Commands
- Use Python 3.11 (accepted by `install.sh`'s Python 3.6+ check). The MAS server invokes this repository's `venv/bin/python`; do not create `.venv/` for that integration.
- From this repository root, create the environment with `python3.11 -m venv venv` and install dependencies with `venv/bin/python -m pip install -r requirements.txt pytest`.
- Run all tests with `venv/bin/python -m pytest tests/` (collects the repository's JCE tests).
- Run one test module with `venv/bin/python -m pytest tests/test_version.py`.
- Check the CLI with `venv/bin/python job_card_extractor.py --version`.

## Gotchas
- PDF conversion needs Poppler on the system (`brew install poppler` on macOS; `apt-get install poppler-utils` on Debian/Ubuntu).
- The first real OCR run downloads EasyOCR language models; the unit tests mock OCR and PDF processing.
- `install.sh` installs into `$HOME/.venv/job-card-extractor`, not the MAS server's repo-local `venv/`; use the commands above for server integration.
