#!/usr/bin/env sh
set -eu

if ! command -v uv >/dev/null 2>&1; then
    printf '%s\n' 'Installing uv...'
    curl -LsSf https://astral.sh/uv/install.sh | sh
    PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    export PATH
fi

if ! command -v uv >/dev/null 2>&1; then
    printf '%s\n' 'Could not find uv after installation.' >&2
    printf '%s\n' 'Open a new terminal and run this installer again.' >&2
    exit 1
fi

printf '%s\n' 'Installing OpenCLI...'
uv tool install --upgrade git+https://github.com/mnisperuza/OpenCLI.git
printf '%s\n' 'OpenCLI installed. Run: opencli'
