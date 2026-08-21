from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from main.session_memory import SessionMemoryStore


class SessionMemoryStoreTests(TestCase):
    def test_session_is_dated_markdown_and_workspace_scoped(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            workspace = root / "workspace"
            workspace.mkdir()
            store = SessionMemoryStore(workspace, root=root / "sessions")
            record = store.create(datetime(2026, 8, 16, 12, 30, tzinfo=timezone.utc))

            store.save(record, "USER: hello\n\nASSISTANT: hi")

            self.assertTrue(record.path.name.startswith("2026-08-16_12-30-00_"))
            self.assertEqual(store.list(), [record.path])
            self.assertIn("USER: hello", store.load(record.path))

    def test_explicit_note_is_preserved_when_transcript_updates(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            workspace = root / "workspace"
            workspace.mkdir()
            store = SessionMemoryStore(workspace, root=root / "sessions")
            record = store.create()

            store.remember(record, "Use Python 3.12", "USER: first")
            store.save(record, "USER: second")
            archived = store.load(record.path)

            self.assertIn("- Use Python 3.12", archived)
            self.assertIn("USER: second", archived)
