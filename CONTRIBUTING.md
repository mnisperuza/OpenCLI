# Contributing to Fenrir Agent

Thanks for improving Fenrir Agent. Keep changes focused, secure, and usable from a
real terminal.

## Development setup

Fenrir Agent supports Python 3.10–3.12. Create an isolated environment, install the
package in editable mode, and run the complete test suite:

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[enterprise]" "pytest>=8,<10"
python -m pytest -q
```

The normal install is intentionally sufficient for hosted-provider work. Local
GGUF use additionally needs `llama-server` from llama.cpp. Do not commit model
weights, credentials, local sessions, or generated logs.

## Pull requests

- Add or update tests for behavior changes.
- Keep tool permissions, workspace boundaries, and untrusted-data handling
  intact. A convenience feature must not silently expand agent authority.
- Run `python -m compileall -q fenrir_agent` and `python -m pytest -q` before
  opening a pull request.
- Describe user-facing behavior, validation performed, and any platform impact.

The release workflow runs Python 3.10–3.12 on Linux, Windows, and macOS for
harness gates, plus the full regression suite on Linux.

## Design boundary

`fenrir_agent` is the implementation and public Python namespace. New
integrations should extend its contracts in `fenrir_agent.interfaces`; UI code
must not gain direct dependencies on a specific model provider. Keep optional
provider dependencies optional, and preserve the local llama.cpp default.

## Commit and review standard

- Use focused, imperative commit subjects such as `fix: reject cross-origin API redirects`.
- Keep generated artifacts, model weights, secrets, local `.fenrir` state, and virtual
  environments out of commits.
- Include a concise test note in every pull request. Security- or sandbox-related
  changes must include a regression test or explain why one is not feasible.
