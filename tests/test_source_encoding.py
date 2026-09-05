"""Regression checks for user-visible source encoding."""

from pathlib import Path
import unittest


class SourceEncodingTests(unittest.TestCase):
    def test_runtime_sources_do_not_contain_common_mojibake(self):
        root = Path(__file__).resolve().parents[1]
        markers = ("â", "Â", "Ã", "ðŸ")

        for relative_path in ("fenrir_agent/cli.py", "fenrir_agent/engine.py"):
            source = (root / relative_path).read_text(encoding="utf-8")
            found = [marker for marker in markers if marker in source]
            self.assertFalse(
                found,
                f"{relative_path} contains encoding artifacts: {found}",
            )


if __name__ == "__main__":
    unittest.main()
