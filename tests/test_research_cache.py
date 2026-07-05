import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from bot import metaculus_bot as metaculus_bot_module
from bot.research_cache import (
    ResearchCache,
    ResearchCacheConfig,
    build_research_cache_key,
    research_cache_options_from_env,
)


class TestResearchCache(unittest.TestCase):
    def test_key_changes_when_resolution_criteria_changes(self) -> None:
        q1 = SimpleNamespace(
            page_url="https://example.com/q/1",
            question_text="Will X happen?",
            background_info="",
            resolution_criteria="Resolve yes if X happens.",
            fine_print="",
            close_time=None,
            scheduled_resolution_time=None,
        )
        q2 = SimpleNamespace(**{**q1.__dict__, "resolution_criteria": "Changed."})

        key1 = build_research_cache_key(
            question=q1, researcher_name="tavily-searcher/kiconnect"
        )
        key2 = build_research_cache_key(
            question=q2, researcher_name="tavily-searcher/kiconnect"
        )

        self.assertNotEqual(key1, key2)

    def test_cache_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = ResearchCache(
                ResearchCacheConfig(
                    enabled=True,
                    path=Path(tmp) / "research_cache.json",
                    ttl_hours=24,
                    max_entries=10,
                )
            )
            cache.set("abc", "research text")

            self.assertEqual(cache.get("abc"), "research text")

    def test_cache_respects_ttl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = ResearchCache(
                ResearchCacheConfig(
                    enabled=True,
                    path=Path(tmp) / "research_cache.json",
                    ttl_hours=0.000001,
                    max_entries=10,
                )
            )
            cache.set("abc", "research text")
            time.sleep(0.01)

            self.assertIsNone(cache.get("abc"))

    def test_disabled_cache_does_not_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "research_cache.json"
            cache = ResearchCache(
                ResearchCacheConfig(
                    enabled=False,
                    path=path,
                    ttl_hours=24,
                    max_entries=10,
                )
            )
            cache.set("abc", "research text")

            self.assertIsNone(cache.get("abc"))
            self.assertFalse(path.exists())


class TestResearchCacheIntegration(unittest.IsolatedAsyncioTestCase):
    async def test_run_research_cache_hit_skips_retrieval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "research_cache.json"
            previous_env = {
                "BOT_ENABLE_RESEARCH_CACHE": None,
                "BOT_RESEARCH_CACHE_PATH": None,
                "BOT_RESEARCH_CACHE_TTL_HOURS": None,
                "BOT_ENABLE_TOOL_TRACE": None,
            }
            import os

            for key in previous_env:
                previous_env[key] = os.environ.get(key)
            try:
                os.environ["BOT_ENABLE_RESEARCH_CACHE"] = "true"
                os.environ["BOT_RESEARCH_CACHE_PATH"] = str(path)
                os.environ["BOT_RESEARCH_CACHE_TTL_HOURS"] = "24"
                os.environ["BOT_ENABLE_TOOL_TRACE"] = "false"

                question = SimpleNamespace(
                    page_url="https://example.com/q/1",
                    question_text="Will X happen?",
                    background_info="",
                    resolution_criteria="Resolve yes if X happens.",
                    fine_print="",
                    close_time=None,
                    scheduled_resolution_time=None,
                )
                key = build_research_cache_key(
                    question=question,
                    researcher_name="free/test",
                    options=research_cache_options_from_env(),
                )
                ResearchCache().set(key, "cached research")

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
                        "researcher": "free/test",
                    },
                )

                async def fail_if_called(_question):
                    raise AssertionError("cache hit should skip retrieval")

                bot._get_local_crawl_context_cached = fail_if_called

                self.assertEqual(await bot.run_research(question), "cached research")
            finally:
                for key, value in previous_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
