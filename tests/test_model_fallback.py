import unittest

import litellm

from bot.metaculus_bot import MetaculusBot


class ModelFallbackTests(unittest.TestCase):
    def test_missing_model_tries_the_next_model(self) -> None:
        # Shape of the KIconnect error seen in Actions from 2026-09-21.
        error = litellm.NotFoundError(
            message="OpenAIException - The model 'qwen3.8-27b' does not exist.",
            model="qwen3.8-27b",
            llm_provider="openai",
        )

        self.assertTrue(MetaculusBot._should_try_fallback_model(error))

    def test_transient_errors_still_fall_back(self) -> None:
        self.assertTrue(MetaculusBot._should_try_fallback_model(TimeoutError("timed out")))

    def test_bad_requests_do_not_fall_back(self) -> None:
        error = litellm.BadRequestError(
            message="OpenAIException - invalid prompt",
            model="gpt-oss-120b",
            llm_provider="openai",
        )

        self.assertFalse(MetaculusBot._should_try_fallback_model(error))


if __name__ == "__main__":
    unittest.main()
