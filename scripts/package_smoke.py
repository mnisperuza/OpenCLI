"""Install a built wheel into a clean environment and verify the CLI entry point."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


def environment_python(environment: Path) -> Path:
    """Return the virtual-environment interpreter for the current platform."""
    return environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def environment_command(environment: Path, name: str) -> Path:
    """Return an installed console-script path for the current platform."""
    return environment / (
        f"Scripts/{name}.exe" if sys.platform == "win32" else f"bin/{name}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packages_dir", type=Path, help="Directory containing built wheels")
    args = parser.parse_args()

    packages_dir = args.packages_dir.resolve()
    wheels = sorted(packages_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError(f"Expected exactly one wheel in {packages_dir}, found {len(wheels)}")

    with tempfile.TemporaryDirectory(prefix="fenrir-agent-wheel-smoke-") as directory:
        environment = Path(directory) / "venv"
        subprocess.run([sys.executable, "-m", "venv", str(environment)], check=True)
        python = environment_python(environment)
        subprocess.run(
            [str(python), "-m", "pip", "install", "--disable-pip-version-check", str(wheels[0])],
            check=True,
        )
        subprocess.run([str(python), "-m", "fenrir_agent", "--version"], check=True)
        subprocess.run(
            [str(environment_command(environment, "fenrir")), "--version"],
            check=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
