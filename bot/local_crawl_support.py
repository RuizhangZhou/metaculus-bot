import asyncio
import logging
import os
from typing import Sequence

from bot.env import env_bool as _env_bool, env_int as _env_int

from local_web_crawl import (
    LocalCrawlLimits,
    PlaywrightWebPageParser,
    crawl_urls,
    extract_http_urls,
)

from forecasting_tools import ForecastReport, MetaculusQuestion

logger = logging.getLogger(__name__)

_NOTEPAD_LOCAL_CRAWL_TASK_KEY = "local_crawl_context_task"


class LocalCrawlSupportMixin:
    _local_crawl_parser: PlaywrightWebPageParser | None = None
    _local_crawl_limits: LocalCrawlLimits | None = None

    def _local_crawl_enabled(self) -> bool:
        return _env_bool("BOT_ENABLE_LOCAL_QUESTION_CRAWL", False)

    def _build_local_crawl_limits_from_env(self) -> LocalCrawlLimits:
        blocked_raw = os.getenv(
            "BOT_LOCAL_CRAWL_BLOCKED_RESOURCE_TYPES", "image,font,media"
        )
        blocked = frozenset(
            {
                part.strip().lower()
                for part in blocked_raw.split(",")
                if part.strip()
            }
        )
        return LocalCrawlLimits(
            max_urls=_env_int("BOT_LOCAL_CRAWL_MAX_URLS", 8),
            max_concurrency=_env_int("BOT_LOCAL_CRAWL_MAX_CONCURRENCY", 2),
            navigation_timeout_seconds=_env_int(
                "BOT_LOCAL_CRAWL_NAV_TIMEOUT_SECONDS", 30
            ),
            network_idle_timeout_seconds=_env_int(
                "BOT_LOCAL_CRAWL_NETWORK_IDLE_TIMEOUT_SECONDS", 5
            ),
            total_char_budget=_env_int(
                "BOT_LOCAL_CRAWL_TOTAL_CHAR_BUDGET", 20_000
            ),
            per_url_char_budget=_env_int(
                "BOT_LOCAL_CRAWL_PER_URL_CHAR_BUDGET", 4_000
            ),
            truncation_marker=os.getenv(
                "BOT_LOCAL_CRAWL_TRUNCATION_MARKER", "\n\n[TRUNCATED]"
            ),
            blocked_resource_types=blocked or frozenset(),
            allow_private_hosts=_env_bool(
                "BOT_LOCAL_CRAWL_ALLOW_PRIVATE_HOSTS", False
            ),
            ignore_https_errors=_env_bool(
                "BOT_LOCAL_CRAWL_IGNORE_HTTPS_ERRORS", False
            ),
            resolve_dns=_env_bool("BOT_LOCAL_CRAWL_RESOLVE_DNS", True),
        )

    async def forecast_questions(
        self,
        questions: Sequence[MetaculusQuestion],
        return_exceptions: bool = False,
    ) -> list[ForecastReport] | list[ForecastReport | BaseException]:
        if not self._local_crawl_enabled():
            return await super().forecast_questions(questions, return_exceptions)

        limits = self._build_local_crawl_limits_from_env()
        user_agent = os.getenv("BOT_LOCAL_CRAWL_USER_AGENT", "").strip() or None

        self._local_crawl_limits = limits
        self._local_crawl_parser = PlaywrightWebPageParser(
            limits=limits,
            user_agent=user_agent,
        )
        try:
            return await super().forecast_questions(questions, return_exceptions)
        finally:
            try:
                await self._local_crawl_parser.close()
            finally:
                self._local_crawl_parser = None
                self._local_crawl_limits = None

    async def _get_local_crawl_context(self, question: MetaculusQuestion) -> str:
        if not self._local_crawl_enabled():
            return ""
        if self._local_crawl_parser is None or self._local_crawl_limits is None:
            return ""

        include_question_page = _env_bool(
            "BOT_LOCAL_CRAWL_INCLUDE_QUESTION_PAGE", True
        )

        urls: list[str] = []
        if include_question_page and getattr(question, "page_url", None):
            urls.append(str(question.page_url))

        urls.extend(extract_http_urls(question.question_text or ""))
        urls.extend(extract_http_urls(question.background_info or ""))
        urls.extend(extract_http_urls(question.resolution_criteria or ""))
        urls.extend(extract_http_urls(question.fine_print or ""))

        # If there is remaining crawl capacity, augment with a small number of curated catalog URLs.
        try:
            _, catalog_urls = self._get_source_catalog_suggestions(question=question)
            max_extra = max(0, int(self._source_catalog_crawl_max_urls()))
            if max_extra > 0 and catalog_urls:
                urls.extend(list(catalog_urls)[:max_extra])
        except Exception:
            pass

        try:
            return await crawl_urls(
                parser=self._local_crawl_parser,
                urls=urls,
                limits=self._local_crawl_limits,
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if isinstance(e, RuntimeError) and "Playwright is not installed" in str(e):
                logger.warning(str(e))
                return ""
            logger.info(
                f"Local crawl failed for question {question.page_url}: {e}"
            )
            return ""

    async def _get_local_crawl_context_cached(
        self, question: MetaculusQuestion
    ) -> str:
        if not self._local_crawl_enabled():
            return ""

        try:
            notepad = await self._get_notepad(question)
        except Exception:
            return await self._get_local_crawl_context(question)

        existing = notepad.note_entries.get(_NOTEPAD_LOCAL_CRAWL_TASK_KEY)
        if isinstance(existing, str):
            return existing
        if isinstance(existing, asyncio.Task):
            try:
                return await existing
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.debug("Local crawl cached task failed", exc_info=True)
                notepad.note_entries[_NOTEPAD_LOCAL_CRAWL_TASK_KEY] = ""
                return ""

        task = asyncio.create_task(self._get_local_crawl_context(question))
        notepad.note_entries[_NOTEPAD_LOCAL_CRAWL_TASK_KEY] = task
        try:
            result = await task
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug("Local crawl task failed", exc_info=True)
            result = ""

        notepad.note_entries[_NOTEPAD_LOCAL_CRAWL_TASK_KEY] = result
        return result

