import unittest

from forecasting_tools import GeneralLlm

from secret_redaction import redact_secrets
from template_bot_2026 import SpringTemplateBot2026


class TestSecretRedaction(unittest.TestCase):
    def test_redact_secrets_handles_nested_headers(self) -> None:
        data = {
            "extra_headers": {
                "Authorization": "Bearer super-secret",
                "X-API-Key": "also-secret",
            },
            "safe": 123,
        }
        redacted = redact_secrets(data)
        self.assertEqual(redacted["extra_headers"]["Authorization"], "<redacted>")
        self.assertEqual(redacted["extra_headers"]["X-API-Key"], "<redacted>")
        self.assertEqual(redacted["safe"], 123)

    def test_bot_make_llm_dict_redacts_api_key(self) -> None:
        secret = "unit-test-keyId:unit-test-secret"
        bot = SpringTemplateBot2026(
            research_reports_per_question=1,
            predictions_per_research_report=1,
            use_research_summary_to_forecast=False,
            publish_reports_to_metaculus=False,
            folder_to_save_reports_to=None,
            skip_previously_forecasted_questions=False,
            extra_metadata_in_explanation=True,
            llms={
                "default": GeneralLlm(
                    model="openai/gpt-4o-mini",
                    base_url="https://chat.kiconnect.nrw/api/v1",
                    api_key=secret,
                    timeout=1,
                )
            },
        )
        llm_dict = bot.make_llm_dict()
        default = llm_dict.get("default")
        self.assertIsInstance(default, str)
        self.assertEqual(default, "openai/gpt-4o-mini")


if __name__ == "__main__":
    unittest.main()
