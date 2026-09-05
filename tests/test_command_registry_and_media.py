from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from PIL import Image
from unittest.mock import patch

from fenrir_agent.cli import FenrirAgent
from fenrir_agent.command_registry import COMMAND_SPECS, match_commands
from fenrir_agent.media import MediaError, load_model_image, normalize_image


class CommandRegistryTests(TestCase):
    def test_commands_are_unique_and_slash_prefixed(self):
        commands = [spec.command for spec in COMMAND_SPECS]
        self.assertEqual(len(commands), len(set(commands)))
        self.assertTrue(all(command.startswith("/") for command in commands))

    def test_match_returns_existing_command_metadata(self):
        matches = match_commands("/cont")
        self.assertEqual(matches[0].command, "/context")
        self.assertIn("context", matches[0].description.casefold())

    def test_search_variants_are_available_to_slash_autocomplete(self):
        matches = match_commands("/search")
        self.assertEqual(
            [item.completion for item in matches],
            ["/search fast", "/search deep", "/search status"],
        )
        self.assertEqual(match_commands("/search d")[0].completion, "/search deep")

    def test_harness_modes_are_available_to_slash_autocomplete(self):
        matches = match_commands("/harness mode")
        self.assertEqual(
            [item.completion for item in matches],
            ["/harness mode legacy", "/harness mode v2"],
        )

    def test_info_command_is_registered_and_dispatched(self):
        self.assertIn("/info", [spec.command for spec in COMMAND_SPECS])
        cli = FenrirAgent(dry_run=True)
        with patch.object(cli, "show_info") as show_info:
            self.assertTrue(cli.handle_command("/info"))
        show_info.assert_called_once()

    def test_mistyped_slash_command_is_consumed_without_model_turn(self):
        cli = FenrirAgent(dry_run=True)
        with patch("builtins.print") as output:
            self.assertTrue(cli.handle_command("/serach deep"))
        self.assertIn("Did you mean `/search fast|deep|status`?", output.call_args.args[0])

    def test_known_command_with_invalid_arguments_shows_usage(self):
        cli = FenrirAgent(dry_run=True)
        with patch("builtins.print") as output:
            self.assertTrue(cli.handle_command("/status extra"))
        self.assertEqual(
            output.call_args.args[0], "Invalid command syntax. Usage: /status"
        )

    def test_toolset_commands_rebuild_the_agent_configuration(self):
        cli = FenrirAgent(dry_run=True)
        with patch("builtins.print"):
            self.assertTrue(cli.handle_command("/tools disable web"))
            self.assertNotIn("web", cli.enabled_toolsets)
            self.assertTrue(cli.handle_command("/tools enable web"))
            self.assertIn("web", cli.enabled_toolsets)
            self.assertTrue(cli.handle_command("/tools reset"))
        self.assertIn("workspace", cli.enabled_toolsets)

    def test_search_command_sets_default_mode_and_rebuilds_runtime(self):
        cli = FenrirAgent(dry_run=True)
        cli.agent_runtime = object()
        with patch("builtins.print"):
            self.assertTrue(cli.handle_command("/search deep"))
            self.assertTrue(cli.handle_command("/search status"))
        self.assertEqual(cli.web_search_mode, "deep")
        self.assertIsNone(cli.agent_runtime)


class MediaFoundationTests(TestCase):
    def test_normalize_converts_rgb_and_bounds_dimensions(self):
        image = Image.new("RGBA", (5000, 1000), (10, 20, 30, 128))
        normalized = normalize_image(image)
        self.assertEqual(normalized.mode, "RGB")
        self.assertEqual(normalized.size, (4096, 819))

    def test_load_corrects_exif_orientation(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "rotated.jpg"
            image = Image.new("RGB", (40, 20), "navy")
            exif = image.getexif()
            exif[274] = 6
            image.save(path, exif=exif)
            normalized = load_model_image(path)
        self.assertEqual(normalized.size, (20, 40))

    def test_corrupt_image_fails_cleanly(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "broken.png"
            path.write_bytes(b"not an image")
            with self.assertRaises(MediaError):
                load_model_image(path)
