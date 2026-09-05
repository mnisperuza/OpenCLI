import tomllib
from pathlib import Path
from unittest import TestCase

from fenrir_agent import __version__
from fenrir_agent.cli import FenrirAgent
import fenrir_agent


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PackagingTests(TestCase):
    def test_cli_uses_package_version(self):
        self.assertEqual(FenrirAgent.VERSION, __version__)

    def test_changelog_current_release_matches_package_version(self):
        content = (PROJECT_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        release_headings = [
            line.removeprefix("## ").strip()
            for line in content.splitlines()
            if line.startswith("## ")
        ]

        self.assertTrue(release_headings)
        self.assertEqual(release_headings[0], __version__)

    def test_pyproject_loads_version_dynamically(self):
        with (PROJECT_ROOT / "pyproject.toml").open("rb") as file:
            metadata = tomllib.load(file)

        project = metadata["project"]
        self.assertNotIn("version", project)
        self.assertIn("version", project["dynamic"])
        self.assertEqual(
            metadata["tool"]["setuptools"]["dynamic"]["version"]["attr"],
            "fenrir_agent._version.__version__",
        )
        self.assertEqual(project["name"], "fenrir-agent")
        self.assertEqual(project["license"], "Apache-2.0")
        self.assertEqual(project["readme"], "README.md")
        self.assertEqual(project["scripts"]["fenrir"], "fenrir_agent.cli:main")
        self.assertIn("mnisperuza/OpenCLI", project["urls"]["Repository"])

    def test_public_library_namespace_exposes_stable_entry_points(self):
        self.assertEqual(fenrir_agent.__version__, __version__)
        self.assertIs(fenrir_agent.FenrirAgent, FenrirAgent)
        self.assertTrue(hasattr(fenrir_agent, "ModelBackend"))

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

        install_command = (
            "uv tool install --upgrade "
            "git+https://github.com/mnisperuza/OpenCLI.git"
        )
        self.assertIn(install_command, shell)
        self.assertIn(install_command, powershell)
        self.assertNotIn("uv tool install --upgrade fenrir-agent", shell)
        self.assertNotIn("uv tool install --upgrade fenrir-agent", powershell)
        self.assertIn("Run: fenrir", shell)
        self.assertIn("Run: fenrir", powershell)
        self.assertIn("astral.sh/uv/install.sh", shell)
        self.assertIn("astral.sh/uv/install.ps1", powershell)

    def test_release_automation_has_tag_and_pypi_trusted_publishing_paths(self):
        workflow = (PROJECT_ROOT / ".github" / "workflows" / "release.yml").read_text(
            encoding="utf-8"
        )
        release_guide = (PROJECT_ROOT / "docs" / "RELEASING.md").read_text(
            encoding="utf-8"
        )

        self.assertIn('tags: ["v*"]', workflow)
        self.assertIn("scripts/verify_release.py", workflow)
        self.assertIn("scripts/package_smoke.py", workflow)
        self.assertIn("pypa/gh-action-pypi-publish@release/v1", workflow)
        self.assertIn("Trusted Publishing", release_guide)
