"""Verify that a release tag names the version packaged by this checkout."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = PROJECT_ROOT / "fenrir_agent" / "_version.py"
VERSION_PATTERN = re.compile(r'^__version__\s*=\s*"(?P<version>[^"\s]+)"\s*$', re.MULTILINE)


def package_version() -> str:
    """Return the one declared package version without importing application code."""
    match = VERSION_PATTERN.search(VERSION_FILE.read_text(encoding="utf-8"))
    if not match:
        raise ValueError("Could not read package version")
    return match.group("version")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="Release tag, for example v2.0.0")
    args = parser.parse_args()

    expected_tag = f"v{package_version()}"
    if args.tag != expected_tag:
        print(
            f"Release tag {args.tag!r} does not match package version {expected_tag!r}.",
            file=sys.stderr,
        )
        return 1
    print(f"Release tag verified: {args.tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
