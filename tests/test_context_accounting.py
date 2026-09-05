from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from fenrir_agent.cli import FenrirAgent
from fenrir_agent.context_accounting import ContextAccountingService, tiktoken_counter
from fenrir_agent.model_profiles import FALLBACK_PROFILE, ModelProfileRegistry


class ModelProfileTests(TestCase):
    def test_builtin_profile_resolves_by_alias_and_model_id(self):
        with TemporaryDirectory() as directory:
            registry = ModelProfileRegistry(Path(directory))
            by_key = registry.resolve(
                key="ministral-3-14b",
                model_id="unused",
                backend="llama_cpp",
            )
            by_id = registry.resolve(
                key="unused",
                model_id="mistralai/Ministral-3-14B-Instruct-2512-GGUF",
                backend="llama_cpp",
            )
        self.assertEqual(by_key, by_id)
        self.assertEqual(by_key.context_window, 32768)
        self.assertTrue(by_key.supports_tools)

    def test_unknown_model_uses_conservative_fallback(self):
        with TemporaryDirectory() as directory:
            profile = ModelProfileRegistry(Path(directory)).resolve(
                key="unknown", model_id="vendor/new-model", backend="remote_api"
            )
        self.assertEqual(profile.context_window, 32768)
        self.assertEqual(profile.max_output_tokens, 4096)
        self.assertFalse(profile.supports_tools)
        self.assertEqual(profile.source, "conservative fallback")

    def test_workspace_override_wins_for_provider_model(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / ".fenrir" / "config.toml"
            config.parent.mkdir()
            config.write_text(
                '[models."openrouter:vendor/model"]\n'
                'context_window = 131072\n'
                'max_output_tokens = 4096\n'
                'supports_tools = true\n',
                encoding="utf-8",
            )
            profile = ModelProfileRegistry(root).resolve(
                key="vendor/model",
                model_id="vendor/model",
                backend="remote_api",
                provider="openrouter",
            )
        self.assertEqual(profile.context_window, 131072)
        self.assertEqual(profile.max_output_tokens, 4096)
        self.assertTrue(profile.supports_tools)
        self.assertIn("workspace override", profile.source)


class ContextAccountingTests(TestCase):
    def test_tiktoken_counter_uses_known_encoding(self):
        encoding = SimpleNamespace(
            encode=lambda text, disallowed_special=(): list(text.split())
        )
        fake_tiktoken = SimpleNamespace(
            get_encoding=lambda _name: encoding,
            encoding_for_model=lambda _model: encoding,
        )
        with patch.dict(sys.modules, {"tiktoken": fake_tiktoken}):
            counter = tiktoken_counter("model", "known")
        self.assertIsNotNone(counter)
        self.assertEqual(counter("three exact tokens"), 3)

    def test_snapshot_reserves_output_and_breaks_down_components(self):
        service = ContextAccountingService(FALLBACK_PROFILE)
        snapshot = service.snapshot({"instructions": "abcd", "history": "abcdefgh"})
        self.assertEqual(snapshot.components["instructions"], 1)
        self.assertEqual(snapshot.components["history"], 2)
        self.assertEqual(snapshot.used_tokens, 3)
        self.assertEqual(snapshot.output_reserve, 4096)
        self.assertTrue(snapshot.estimated)

    def test_oversized_output_capability_cannot_consume_clean_input_budget(self):
        profile = replace(
            FALLBACK_PROFILE,
            context_window=32768,
            max_output_tokens=32767,
        )
        snapshot = ContextAccountingService(profile).snapshot(
            {"instructions": "short", "current prompt": "hello"}
        )

        self.assertEqual(snapshot.output_reserve, 16384)
        self.assertGreater(snapshot.available_tokens, 16000)

    def test_local_metadata_caps_output_to_half_context(self):
        with TemporaryDirectory() as directory:
            profile = ModelProfileRegistry(Path(directory)).resolve(
                key="custom",
                model_id="vendor/custom-gguf",
                backend="llama_cpp",
                metadata={"context": 32768, "max_tokens": 32767},
            )

        self.assertEqual(profile.context_window, 32768)
        self.assertEqual(profile.max_output_tokens, 16384)

    def test_tokenizer_and_session_usage_are_exact_when_reported(self):
        service = ContextAccountingService(FALLBACK_PROFILE, tokenizer=len)
        snapshot = service.snapshot({"history": "four"})
        service.record_turn(10, 5, estimated=False)
        self.assertEqual(snapshot.used_tokens, 4)
        self.assertFalse(snapshot.estimated)
        self.assertEqual(service.usage.total_tokens, 15)
        service.reset_usage()
        self.assertEqual(service.usage.turns, 0)

    def test_cli_exposes_context_commands_and_status_indicator(self):
        cli = FenrirAgent(dry_run=True)
        self.assertIn("ctx", cli.context_bar())
        with (
            patch.object(cli, "show_context") as show_context,
            patch.object(cli, "show_usage") as show_usage,
            patch.object(cli, "show_prompt_size") as show_prompt_size,
        ):
            self.assertTrue(cli.handle_command("/context"))
            self.assertTrue(cli.handle_command("/usage"))
            self.assertTrue(cli.handle_command("/prompt-size"))
        show_context.assert_called_once()
        show_usage.assert_called_once()
        show_prompt_size.assert_called_once()

    def test_cli_starts_in_local_llama_cpp_workflow(self):
        cli = FenrirAgent(dry_run=True)

        self.assertEqual(cli.mode, "auto")
        self.assertIsNone(cli.api_provider)
        self.assertEqual(cli.context_accounting.profile.backend, "llama_cpp")

    def test_local_override_updates_engine_context_before_load(self):
        class Engine:
            MODELS = {
                "auto": {
                    "path": "mistralai/Ministral-3-14B-Instruct-2512-GGUF",
                    "backend": "llama_cpp",
                    "context": 32768,
                    "max_tokens": 8192,
                }
            }

            def __init__(self):
                self.loaded_context = None

            def load_model(self, mode, _quant):
                self.loaded_context = self.MODELS[mode]["context"]
                return True, "loaded"

        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / ".fenrir" / "config.toml"
            config.parent.mkdir()
            config.write_text(
                '[models.auto]\ncontext_window = 24576\nmax_output_tokens = 4096\n',
                encoding="utf-8",
            )
            cli = FenrirAgent(dry_run=True)
            cli.engine = Engine()
            cli.model_profiles = ModelProfileRegistry(root)
            with patch("fenrir_agent.cli.loading_spinner"):
                self.assertTrue(cli.load_model("auto", "int4", show_picker=False))

        self.assertEqual(cli.engine.loaded_context, 24576)

    def test_prompt_size_does_not_initialize_model_engine(self):
        cli = FenrirAgent(dry_run=True)
        with (
            patch.object(cli, "ensure_engine") as ensure_engine,
            patch("builtins.print"),
        ):
            cli.show_prompt_size()
        ensure_engine.assert_not_called()
