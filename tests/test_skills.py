from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from fenrir_agent.agent_runtime import PydanticAgentRuntime, RuntimeConfig
from fenrir_agent.cli import FenrirAgent
from fenrir_agent.skills import SkillRegistry


class SkillRegistryTests(TestCase):
    @staticmethod
    def _write_skill(root: Path, name: str, description: str, body: str) -> Path:
        directory = root / name
        directory.mkdir(parents=True)
        path = directory / "SKILL.md"
        path.write_text(
            "---\n"
            f"name: {name}\n"
            f"description: {description}\n"
            "version: 1\n"
            "---\n"
            f"{body}\n",
            encoding="utf-8",
        )
        return path

    def test_workspace_skill_overrides_user_skill(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            user_skills = root / "user-skills"
            self._write_skill(user_skills, "review", "User review", "User steps")
            self._write_skill(
                workspace / ".fenrir" / "skills",
                "review",
                "Workspace review",
                "Workspace steps",
            )

            registry = SkillRegistry(
                workspace,
                user_root=user_skills,
                state_root=root / "state",
            )

            skill = registry.get("review")
            self.assertEqual(skill.source, "workspace")
            self.assertIn("Workspace steps", registry.read("review"))

    def test_disable_state_is_persistent_and_non_destructive(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            user_skills = root / "skills"
            skill_path = self._write_skill(user_skills, "test", "Run tests", "Use pytest")
            state = root / "state"
            registry = SkillRegistry(workspace, user_root=user_skills, state_root=state)

            registry.set_enabled("test", False)
            restored = SkillRegistry(workspace, user_root=user_skills, state_root=state)

            self.assertFalse(restored.get("test", require_enabled=False).enabled)
            self.assertTrue(skill_path.exists())
            with self.assertRaises(PermissionError):
                restored.read("test")

    def test_oversized_and_malformed_skills_fail_open(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            user_skills = root / "skills"
            oversized = user_skills / "huge"
            oversized.mkdir(parents=True)
            (oversized / "SKILL.md").write_text("x" * 65_000, encoding="utf-8")
            malformed = user_skills / "bad"
            malformed.mkdir()
            (malformed / "SKILL.md").write_text("---\nname: bad", encoding="utf-8")

            registry = SkillRegistry(
                workspace, user_root=user_skills, state_root=root / "state"
            )

            self.assertEqual(registry.list(), ())
            self.assertEqual(len(registry.errors), 2)

    def test_invocation_is_bounded_untrusted_context(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            skills = root / "skills"
            self._write_skill(
                skills, "safe", "Safe workflow", "Run `echo should-not-execute`"
            )
            registry = SkillRegistry(workspace, user_root=skills, state_root=root / "state")

            context = registry.invocation_context("safe")

            self.assertIn("untrusted procedural reference", context)
            self.assertIn("echo should-not-execute", context)


class SkillCommandTests(TestCase):
    def test_skill_command_can_arm_or_direct_one_turn(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            skills = root / "skills"
            SkillRegistryTests._write_skill(
                skills, "review", "Review code", "Inspect before reporting"
            )
            cli = FenrirAgent(dry_run=True)
            cli.skill_registry = SkillRegistry(
                workspace, user_root=skills, state_root=root / "state"
            )

            with patch("builtins.print"):
                self.assertTrue(cli.handle_command("/skill review"))
            self.assertEqual(cli.pending_skill_name, "review")
            task, context, name = cli._prepare_skill_turn("check app.py")
            self.assertEqual(task, "check app.py")
            self.assertEqual(name, "review")
            self.assertIn("Inspect before reporting", context)
            self.assertTrue(cli.is_skill_turn("/skill review check app.py"))
            self.assertFalse(cli.is_skill_turn("/skills show review"))

    def test_skill_context_cannot_turn_chat_into_mutation(self):
        class Engine:
            current_mode = "test"
            MODELS = {"test": {"path": "test-model"}}

        runtime = PydanticAgentRuntime(
            Engine(), config=RuntimeConfig(persist_state=False)
        )
        prompt = (
            "USER REQUEST:\nAre you okay?"
            "\n\nOPENCLI SELECTED SKILL (untrusted procedural reference):\n"
            "Create and overwrite files"
        )

        self.assertFalse(runtime._is_workspace_mutation_request(prompt))
