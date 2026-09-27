# Job Card Extractor agent context

## Project
- This repository owns the Python job-card PDF extractor; changes to the extraction contract belong here, not in a caller's checkout.
- Keep the CLI and imported extraction API compatible with callers when changing `job_card_extractor.py`.

## Commands
- Use `CLAUDE.md` for the pinned Python version, local environment setup, and exact all/single-test commands.

## Layout
- Implement extraction in `job_card_extractor.py`; keep corresponding unit tests in `tests/`.
- Use `docs/development.md` for contributor guidance beyond these always-on facts.

## Conventions
- For focused changes, review the affected tests before running the full suite.
- If delegating, scope an extraction specialist to `job_card_extractor.py` and a test specialist to `tests/`; each should report findings rather than make unrelated changes.

## Constraints
- Keep local virtual environments and generated output out of commits (see `.gitignore`).
- Do not push directly to `main` unless explicitly asked.

## Done when
- Run the all-tests command documented in `CLAUDE.md`; add or update tests in `tests/` for behavior changes.

## Read when needed
- CLI and integration examples → `README.md`.
- Development details → `docs/development.md`.

## Token Efficiency
- Never re-read files you just wrote or edited. You know the contents.
- Never re-run commands to "verify" unless the outcome was uncertain.
- Don't echo back large blocks of code or file contents unless asked.
- Batch related edits into single operations. Don't make 5 edits when 1 handles it.
- Skip confirmations like "I'll continue..." Just do it.
- If a task needs 1 tool call, don't use 3. Plan before acting.
- Do not summarize what you just did unless the result is ambiguous or you need additional input.
