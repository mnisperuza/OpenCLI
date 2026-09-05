from unittest import TestCase

from pydantic_ai.messages import ModelRequest, UserPromptPart

from fenrir_agent.agent_runtime import LocalModelAdapter
from fenrir_agent.cli import FenrirAgent
from fenrir_agent.language import response_language


class LanguageGuardTests(TestCase):
    def test_english_prompt_requires_english_response(self):
        prompt = FenrirAgent._model_input("Inspect this repository and explain the tests.")

        self.assertEqual(response_language("Inspect this repository and explain the tests."), "English")
        self.assertIn("RESPONSE LANGUAGE: English", prompt)
        self.assertIn("USER REQUEST", prompt)

    def test_spanish_prompt_requires_spanish_response(self):
        prompt = FenrirAgent._model_input("Revisa este repositorio y explica las pruebas.")

        self.assertEqual(response_language("Revisa este repositorio y explica las pruebas."), "Spanish")
        self.assertIn("RESPONSE LANGUAGE: Spanish", prompt)

    def test_technical_ambiguous_prompt_defaults_to_english(self):
        self.assertEqual(response_language("pytest -q main.py"), "English")

    def test_final_rule_repeats_latest_language_after_tool_rounds(self):
        message = ModelRequest(
            parts=[UserPromptPart(content=FenrirAgent._model_input("Explain this test."))]
        )

        rule = LocalModelAdapter._final_language_rule([message])

        self.assertIn("RESPONSE LANGUAGE: English", rule)
        self.assertIn("Tool and web output", rule)
