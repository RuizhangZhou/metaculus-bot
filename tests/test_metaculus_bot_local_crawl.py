import asyncio
from types import SimpleNamespace

from bot import metaculus_bot as metaculus_bot_module


def test_forecast_questions_starts_and_closes_local_crawl_parser(monkeypatch) -> None:
    events: list[str] = []

    class FakeParser:
        def __init__(self, *, limits, user_agent=None) -> None:
            events.append("started")
            self.limits = limits
            self.user_agent = user_agent

        async def close(self) -> None:
            events.append("closed")

    monkeypatch.setattr(
        metaculus_bot_module, "PlaywrightWebPageParser", FakeParser
    )
    monkeypatch.setenv("BOT_ENABLE_LOCAL_QUESTION_CRAWL", "true")
    monkeypatch.setenv("BOT_MAX_CONCURRENT_QUESTIONS", "1")

    bot = metaculus_bot_module.MetaculusBot(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=False,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=False,
        llms={
            "default": "no_research",
            "summarizer": "no_research",
            "parser": "no_research",
            "researcher": "no_research",
        },
    )

    async def fake_run_question(question):
        assert bot._local_crawl_parser is not None
        assert bot._local_crawl_limits is not None
        return "ok"

    bot._run_individual_question_with_error_propagation = fake_run_question

    reports = asyncio.run(
        bot.forecast_questions([SimpleNamespace(already_forecasted=False)])
    )

    assert reports == ["ok"]
    assert events == ["started", "closed"]
    assert bot._local_crawl_parser is None
    assert bot._local_crawl_limits is None
