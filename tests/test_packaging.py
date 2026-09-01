import tomllib
from pathlib import Path
from unittest import TestCase

from main import __version__
from main.cli import OpenCLI
import opencli


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
        self.assertEqual(project["name"], "opencli")
        self.assertEqual(project["license"], "Apache-2.0")
        self.assertEqual(project["scripts"]["opencli"], "opencli.cli:main")
        self.assertTrue(all("OpenCLI" in url for url in project["urls"].values()))

    def test_public_library_namespace_exposes_stable_entry_points(self):
        self.assertEqual(opencli.__version__, __version__)
        self.assertIs(opencli.OpenCLI, OpenCLI)
        self.assertTrue(hasattr(opencli, "ModelBackend"))

    def test_legacy_site_is_not_shipped_as_product_documentation(self):
        for name in ("index.html", "docs.html", "about.html", "privacy.html", "terms.html"):
            self.assertFalse((PROJECT_ROOT / name).exists(), name)

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

    def test_cross_platform_install_wrappers_use_isolated_tool_install(self):
        shell = (PROJECT_ROOT / "scripts" / "install.sh").read_text(
            encoding="utf-8"
        )
        powershell = (PROJECT_ROOT / "scripts" / "install.ps1").read_text(
            encoding="utf-8"
        )

        self.assertIn("uv tool install --upgrade opencli", shell)
        self.assertIn("uv tool install --upgrade opencli", powershell)
        self.assertIn("astral.sh/uv/install.sh", shell)
        self.assertIn("astral.sh/uv/install.ps1", powershell)
