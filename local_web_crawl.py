from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Iterable

logger = logging.getLogger(__name__)


_URL_CANDIDATE_RE = re.compile(r"https?://[^\s<>\"]+", re.IGNORECASE)
_TRAILING_PUNCTUATION_RE = re.compile(r"[)\].,;:!?]+$")


def extract_http_urls(text: str) -> list[str]:
    if not text:
        return []
    candidates = _URL_CANDIDATE_RE.findall(text)
    cleaned: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        url = _TRAILING_PUNCTUATION_RE.sub("", candidate.strip())
        if not url:
            continue
        if url in seen:
            continue
        seen.add(url)
        cleaned.append(url)
    return cleaned


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def truncate_text(text: str, max_chars: int, marker: str) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    marker = marker or ""
    if len(marker) >= max_chars:
        return marker[:max_chars]
    return text[: max_chars - len(marker)] + marker


@dataclass(frozen=True)
class LocalCrawlLimits:
    max_urls: int = 8
    max_concurrency: int = 2
    navigation_timeout_seconds: int = 30
    network_idle_timeout_seconds: int = 5
    total_char_budget: int = 20_000
    per_url_char_budget: int = 4_000
    truncation_marker: str = "\n\n[TRUNCATED]"
    blocked_resource_types: frozenset[str] = frozenset({"image", "font", "media"})


class PlaywrightWebPageParser:
    def __init__(
        self,
        *,
        limits: LocalCrawlLimits,
        user_agent: str | None = None,
    ) -> None:
        self._limits = limits
        self._user_agent = user_agent
        self._sem = asyncio.Semaphore(max(1, limits.max_concurrency))
        self._pw = None
        self._browser = None
        self._launch_lock = asyncio.Lock()

    async def _ensure_browser(self) -> None:
        if self._browser is not None and self._pw is not None:
            return

        async with self._launch_lock:
            if self._browser is not None and self._pw is not None:
                return

            try:
                from playwright.async_api import async_playwright
            except Exception as e:
                raise RuntimeError(
                    "Playwright is not installed. Install with `poetry install --with web` and run `poetry run playwright install chromium`."
                ) from e

            self._pw = await async_playwright().start()
            self._browser = await self._pw.chromium.launch(headless=True)

    async def close(self) -> None:
        browser = self._browser
        pw = self._pw
        self._browser = None
        self._pw = None

        if browser is not None:
            try:
                await browser.close()
            except Exception:
                logger.exception("Failed to close Playwright browser cleanly")
        if pw is not None:
            try:
                await pw.stop()
            except Exception:
                logger.exception("Failed to stop Playwright cleanly")

    async def get_clean_text(self, url: str) -> str:
        await self._ensure_browser()
        assert self._browser is not None

        async with self._sem:
            context = await self._browser.new_context(
                user_agent=self._user_agent,
                java_script_enabled=True,
                ignore_https_errors=True,
            )
            try:
                await self._apply_resource_blocking(context)
                page = await context.new_page()
                try:
                    page.set_default_navigation_timeout(
                        self._limits.navigation_timeout_seconds * 1000
                    )
                    page.set_default_timeout(
                        self._limits.navigation_timeout_seconds * 1000
                    )
                    await page.goto(url, wait_until="domcontentloaded")
                    try:
                        await page.wait_for_load_state(
                            "networkidle",
                            timeout=self._limits.network_idle_timeout_seconds * 1000,
                        )
                    except Exception:
                        pass

                    html = await page.content()
                    rendered_text = await page.evaluate(
                        "() => document.body ? document.body.innerText : ''"
                    )
                finally:
                    try:
                        await page.close()
                    except Exception:
                        logger.debug("Failed to close Playwright page", exc_info=True)

                extracted = self._extract_main_text(html) or rendered_text
                cleaned = normalize_text(extracted)
                return truncate_text(
                    cleaned,
                    self._limits.per_url_char_budget,
                    self._limits.truncation_marker,
                )
            finally:
                await context.close()

    async def _apply_resource_blocking(self, context: object) -> None:
        blocked = set(self._limits.blocked_resource_types or [])
        if not blocked:
            return

        async def handle_route(route, request) -> None:  # type: ignore[no-untyped-def]
            try:
                if request.resource_type in blocked:
                    await route.abort()
                else:
                    await route.continue_()
            except Exception:
                try:
                    await route.continue_()
                except Exception:
                    pass

        try:
            await context.route("**/*", handle_route)  # type: ignore[attr-defined]
        except Exception:
            logger.debug("Failed to install resource blocking", exc_info=True)

    @staticmethod
    def _extract_main_text(html: str) -> str:
        if not html:
            return ""
        try:
            from readability import Document  # type: ignore[import-not-found]
        except Exception:
            return ""

        try:
            summary_html = Document(html).summary(html_partial=True)
        except Exception:
            return ""

        try:
            import lxml.html  # type: ignore[import-not-found]
        except Exception:
            return ""

        try:
            root = lxml.html.fromstring(summary_html)
            return root.text_content()
        except Exception:
            return ""


async def crawl_urls(
    *,
    parser: PlaywrightWebPageParser,
    urls: Iterable[str],
    limits: LocalCrawlLimits,
) -> str:
    unique: list[str] = []
    seen: set[str] = set()
    for url in urls:
        url = (url or "").strip()
        if not url or not (url.startswith("http://") or url.startswith("https://")):
            continue
        if url in seen:
            continue
        seen.add(url)
        unique.append(url)
        if len(unique) >= max(0, limits.max_urls):
            break

    if not unique:
        return ""

    async def fetch_one(url: str) -> tuple[str, str | None, Exception | None]:
        try:
            text = await parser.get_clean_text(url)
            return url, text, None
        except asyncio.CancelledError:
            raise
        except Exception as e:
            return url, None, e

    fetched: list[tuple[str, str | None, Exception | None]] = await asyncio.gather(
        *[fetch_one(url) for url in unique]
    )

    results: list[str] = []
    remaining = limits.total_char_budget
    for url, text, err in fetched:
        if remaining <= 0:
            break
        if err is not None:
            logger.info(f"Local crawl failed for {url}: {err}")
            results.append(
                f"Source: {url}\n[ERROR] {err.__class__.__name__}: {err}"
            )
            continue
        if not text:
            continue
        text = truncate_text(text, min(len(text), remaining), "")
        remaining -= len(text)
        results.append(f"Source: {url}\n{text}")

    return "\n\n---\n\n".join(results).strip()
