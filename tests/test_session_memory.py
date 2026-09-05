from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from fenrir_agent.session_memory import SessionMemoryStore


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
            store.remember(record, "Use Python 3.12", "USER: first")
            store.save(record, "USER: second")
            archived = store.load(record.path)

            self.assertIn("- Use Python 3.12", archived)
            self.assertEqual(archived.count("- Use Python 3.12"), 1)
            self.assertIn("USER: second", archived)

    def test_context_load_uses_notes_checkpoint_and_recent_transcript(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            workspace = root / "workspace"
            workspace.mkdir()
            store = SessionMemoryStore(workspace, root=root / "sessions")
            record = store.create()
            store.remember(record, "Use Python 3.12", "USER: old")
            store.record_compaction(
                record,
                summary="USER: Keep tests focused.",
                source_transcript="USER: obsolete secret detail",
                transcript="USER: current task\n\nASSISTANT: current answer",
            )

            context = store.load_context(record.path)

        self.assertIn("Use Python 3.12", context)
        self.assertIn("Keep tests focused", context)
        self.assertIn("current task", context)
        self.assertNotIn("obsolete secret detail", context)

    def test_pruned_tool_archive_stays_out_of_loaded_model_context(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            workspace = root / "workspace"
            workspace.mkdir()
            store = SessionMemoryStore(workspace, root=root / "sessions")
            record = store.create()
            store.archive_tool_results(record, "FULL TOOL PAYLOAD " * 100)
            store.save(record, "USER: recent")

            archive = store.load(record.path)
            context = store.load_context(record.path)

        self.assertIn("FULL TOOL PAYLOAD", archive)
        self.assertNotIn("FULL TOOL PAYLOAD", context)

    def test_error_payloads_are_excluded_from_archive_and_loaded_context(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            workspace = root / "workspace"
            workspace.mkdir()
            store = SessionMemoryStore(workspace, root=root / "sessions")
            record = store.create()
            transcript = (
                "USER: inspect the project\n\n"
                "TOOL RESULT [read_text_file] (call 1): "
                '{"error": "Not a file: missing.md"}\n\n'
                "TOOL VALIDATION ERROR: bad arguments\n\n"
                "ASSISTANT: I used the available evidence."
            )

            store.save(record, transcript)
            archive = store.load(record.path)
            context = store.load_context(record.path)

        self.assertNotIn("Not a file", archive)
        self.assertNotIn("VALIDATION ERROR", archive)
        self.assertNotIn("Not a file", context)
        self.assertIn("ASSISTANT: I used the available evidence.", context)

    def test_tool_archive_is_bounded(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            workspace = root / "workspace"
            workspace.mkdir()
            store = SessionMemoryStore(workspace, root=root / "sessions")
            store.MAX_TOOL_ARCHIVE_CHARS = 20
            store.MAX_TOOL_ARCHIVES = 2
            record = store.create()

            store.archive_tool_results(record, "first")
            store.archive_tool_results(record, "x" * 30)
            store.archive_tool_results(record, "latest")

        self.assertEqual(len(record.tool_archives), 2)
        self.assertIn("Archive truncated", record.tool_archives[0].content)
        self.assertEqual(record.tool_archives[1].content, "latest")

    def test_record_round_trip_preserves_title_and_archives_for_resume(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            workspace = root / "workspace"
            workspace.mkdir()
            store = SessionMemoryStore(workspace, root=root / "sessions")
            record = store.create()
            store.remember(record, "Use focused tests", "USER: first")
            store.record_compaction(
                record,
                summary="Goal: fix preview",
                source_transcript="USER: earlier",
                transcript="USER: recent",
            )
            store.archive_tool_results(record, "tool payload")
            store.set_title(record, "Fix preview crash", "USER: recent")

            restored = store.load_record(record.path)
            capsule = store.load_capsule(record.path)

        self.assertEqual(restored.title, "Fix preview crash")
        self.assertEqual(restored.notes, ["Use focused tests"])
        self.assertEqual(restored.compactions[0].summary, "Goal: fix preview")
        self.assertEqual(restored.tool_archives[0].content, "tool payload")
        self.assertIn("SESSION CAPSULE: Fix preview crash", capsule)

    def test_record_round_trip_preserves_logical_current_directory(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            workspace = root / "workspace"
            workspace.mkdir()
            store = SessionMemoryStore(workspace, root=root / "sessions")
            record = store.create()
            record.current_directory = "src/auth"
            store.save(record, "USER: inspect auth")

            restored = store.load_record(record.path)

        self.assertEqual(restored.current_directory, "src/auth")
