import os
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from bot import metaculus_bot as metaculus_bot_module


def _make_bot(*, skip_previously_forecasted_questions: bool = False):
    return metaculus_bot_module.MetaculusBot(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=False,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=skip_previously_forecasted_questions,
        llms={
            "default": "no_research",
            "summarizer": "no_research",
            "parser": "no_research",
            "researcher": "no_research",
        },
    )


class TestBinaryCalibrationGuardrails(unittest.TestCase):
    def test_missing_community_prediction_does_not_anchor_to_half(self) -> None:
        bot = _make_bot()
        question = SimpleNamespace(
            community_prediction_at_access_time=None,
            scheduled_resolution_time=datetime.now(timezone.utc) + timedelta(days=30),
        )

        calibrated = bot._calibrate_binary_probability(
            question=question,
            p=0.80,
            research="Official source https://example.gov says 12 of 15 criteria passed.",
            reasoning="Evidence is direct and current.",
            context="test",
        )

        self.assertAlmostEqual(calibrated, 0.80)

    def test_available_community_prediction_still_anchors_extreme_deviation(self) -> None:
        bot = _make_bot()
        question = SimpleNamespace(
            community_prediction_at_access_time=0.50,
            scheduled_resolution_time=datetime.now(timezone.utc) + timedelta(days=30),
        )

        calibrated = bot._calibrate_binary_probability(
            question=question,
            p=0.80,
            research="Official source https://example.gov says 12 of 15 criteria passed.",
            reasoning="Evidence is direct and current.",
            context="test",
        )

        self.assertGreater(calibrated, 0.50)
        self.assertLess(calibrated, 0.80)


class TestForecastQuestionGuardrails(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._previous_env = {
            "BOT_ENABLE_LOCAL_QUESTION_CRAWL": os.environ.get(
                "BOT_ENABLE_LOCAL_QUESTION_CRAWL"
            ),
            "BOT_MAX_CONCURRENT_QUESTIONS": os.environ.get(
                "BOT_MAX_CONCURRENT_QUESTIONS"
            ),
        }
        os.environ["BOT_ENABLE_LOCAL_QUESTION_CRAWL"] = "false"
        os.environ["BOT_MAX_CONCURRENT_QUESTIONS"] = "1"

    async def asyncTearDown(self) -> None:
        for key, value in self._previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    async def test_forecast_questions_skips_previously_forecasted_questions(self) -> None:
        bot = _make_bot(skip_previously_forecasted_questions=True)
        seen: list[str] = []

        async def fake_run_question(question):
            seen.append(question.page_url)
            return question.page_url

        bot._run_individual_question_with_error_propagation = fake_run_question

        reports = await bot.forecast_questions(
            [
                SimpleNamespace(already_forecasted=True, page_url="old"),
                SimpleNamespace(already_forecasted=False, page_url="new"),
            ]
        )

        self.assertEqual(seen, ["new"])
        self.assertEqual(reports, ["new"])
