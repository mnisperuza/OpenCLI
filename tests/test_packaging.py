import tomllib
from pathlib import Path
from unittest import TestCase

from main import __version__
from main.cli import OpenCLI


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PackagingTests(TestCase):
    def test_cli_uses_package_version(self):
        self.assertEqual(OpenCLI.VERSION, __version__)

    def test_pyproject_loads_version_dynamically(self):
        with (PROJECT_ROOT / "pyproject.toml").open("rb") as file:
            metadata = tomllib.load(file)

        project = metadata["project"]
        self.assertNotIn("version", project)
        self.assertIn("version", project["dynamic"])
        self.assertEqual(
            metadata["tool"]["setuptools"]["dynamic"]["version"]["attr"],
            "main._version.__version__",
        )

    def test_legacy_requirements_delegate_to_pyproject(self):
        content = (PROJECT_ROOT / "requirements.txt").read_text(
            encoding="utf-8"
        )
        active_lines = [
            line.strip()
            for line in content.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        self.assertEqual(active_lines, ["-e ."])
