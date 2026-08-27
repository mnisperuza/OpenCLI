from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from PIL import Image

from main.command_registry import COMMAND_SPECS, match_commands
from main.media import MediaError, load_model_image, normalize_image


class CommandRegistryTests(TestCase):
    def test_commands_are_unique_and_slash_prefixed(self):
        commands = [spec.command for spec in COMMAND_SPECS]
        self.assertEqual(len(commands), len(set(commands)))
        self.assertTrue(all(command.startswith("/") for command in commands))

    def test_match_returns_existing_command_metadata(self):
        matches = match_commands("/cont")
        self.assertEqual(matches[0].command, "/context")
        self.assertIn("context", matches[0].description.casefold())


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
