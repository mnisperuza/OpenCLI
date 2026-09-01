# Contributing to OpenCLI

Thanks for improving OpenCLI. Keep changes focused, secure, and usable from a
real terminal.

## Development setup

OpenCLI supports Python 3.10–3.12. Create an isolated environment, install the
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
- Run `python -m compileall -q main opencli` and `python -m pytest -q` before
  opening a pull request.
- Describe user-facing behavior, validation performed, and any platform impact.

The release workflow runs Python 3.10–3.12 on Linux, Windows, and macOS for
harness gates, plus the full regression suite on Linux.

## Design boundary

`main` is the implementation namespace. New integrations should use the public
`opencli` package or extend the contracts in `main.interfaces`; UI code should
not gain direct dependencies on a specific model provider.
