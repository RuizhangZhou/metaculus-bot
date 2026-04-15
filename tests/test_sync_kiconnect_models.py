import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.sync_kiconnect_models import (
    TEMPLATE_BLOCK_END,
    TEMPLATE_BLOCK_START,
    build_template_block,
    filter_chat_models,
    prioritize_chat_models,
    update_dotenv_file,
    update_template_file,
)


class TestSyncKiconnectModels(unittest.TestCase):
    def test_filter_chat_models_excludes_embeddings(self) -> None:
        model_ids = [
            "gpt-oss-120b",
            "qwen3-embedding-8b",
            "e5-mistral-7b-instruct",
            "mistral-small-4-119b-2603",
            "gpt-5.2",
            "gpt-5.4-mini",
        ]

        self.assertEqual(
            prioritize_chat_models(filter_chat_models(model_ids)),
            [
                "gpt-5.2",
                "gpt-5.4-mini",
                "gpt-oss-120b",
                "mistral-small-4-119b-2603",
            ],
        )

    def test_update_dotenv_file_replaces_target_keys(self) -> None:
        with TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "KICONNECT_MODEL=gpt-5.1\n"
                "KICONNECT_MODEL_FALLBACKS=gpt-4.1,gpt-4o\n"
                "OTHER=value\n",
                encoding="utf-8",
            )

            changed = update_dotenv_file(
                env_path,
                {
                    "KICONNECT_MODEL": "gpt-5.2",
                    "KICONNECT_MODEL_FALLBACKS": "gpt-5.4-mini,gpt-oss-120b",
                },
                dry_run=False,
            )

            self.assertTrue(changed)
            self.assertEqual(
                env_path.read_text(encoding="utf-8"),
                "KICONNECT_MODEL=gpt-5.2\n"
                "KICONNECT_MODEL_FALLBACKS=gpt-5.4-mini,gpt-oss-120b\n"
                "OTHER=value\n",
            )

    def test_update_template_file_manages_summary_block(self) -> None:
        with TemporaryDirectory() as temp_dir:
            template_path = Path(temp_dir) / ".env.template"
            template_path.write_text(
                "# Optional: Use KIconnect as your non-search LLM (OpenAI-compatible proxy)\n"
                "KICONNECT_API_URL=YOUR_KICONNECT_API_URL\n",
                encoding="utf-8",
            )

            changed = update_template_file(
                template_path,
                ["gpt-5.2", "gpt-5.4-mini", "gpt-oss-120b"],
                "gpt-5.2",
                ["gpt-5.4-mini", "gpt-oss-120b"],
                dry_run=False,
            )

            self.assertTrue(changed)
            content = template_path.read_text(encoding="utf-8")
            self.assertIn(TEMPLATE_BLOCK_START, content)
            self.assertIn(TEMPLATE_BLOCK_END, content)
            self.assertIn("Current KICONNECT chat models: gpt-5.2,gpt-5.4-mini,gpt-oss-120b", content)
            self.assertIn("Recommended KICONNECT_MODEL: gpt-5.2", content)

    def test_build_template_block_handles_empty_fallbacks(self) -> None:
        block = build_template_block(["gpt-5.2"], "gpt-5.2", [])
        self.assertIn("Recommended KICONNECT_MODEL_FALLBACKS: (none)", block)


if __name__ == "__main__":
    unittest.main()
