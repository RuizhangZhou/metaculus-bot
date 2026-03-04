from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime

import aiohttp

from forecasting_tools import GeneralLlm, clean_indents
from forecasting_tools.util.misc import fill_in_citations

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TavilySearchResult:
    query: str
    title: str | None
    url: str | None
    content: str | None
    raw_content: str | None
    score: float | None


class TavilySearcher:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        timeout_seconds: float = 30,
        max_concurrency: int = 3,
        search_depth: str = "basic",
        topic: str = "general",
        time_range: str | None = None,
        include_raw_content: bool = False,
        include_images: bool = False,
    ) -> None:
        self._api_key = (api_key or os.getenv("TAVILY_API_KEY") or "").strip()
        if not self._api_key:
            raise ValueError("TAVILY_API_KEY is not set in the environment variables")

        self._timeout = aiohttp.ClientTimeout(total=max(1.0, float(timeout_seconds)))
        self._sem = asyncio.Semaphore(max(1, int(max_concurrency)))

        self._search_depth = (search_depth or "basic").strip()
        self._topic = (topic or "general").strip()
        self._time_range = (time_range or "").strip() or None
        self._include_raw_content = bool(include_raw_content)
        self._include_images = bool(include_images)

    async def search(
        self,
        *,
        query: str,
        max_results: int,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
    ) -> list[TavilySearchResult]:
        query = (query or "").strip()
        if not query:
            return []

        url = "https://api.tavily.com/search"
        headers = {"Authorization": f"Bearer {self._api_key}"}
        payload: dict = {
            "query": query,
            "search_depth": self._search_depth,
            "topic": self._topic,
            "max_results": max(1, int(max_results)),
            "include_answer": False,
            "include_raw_content": self._include_raw_content,
            "include_images": self._include_images,
            "include_domains": include_domains or [],
            "exclude_domains": exclude_domains or [],
        }
        if self._time_range:
            payload["time_range"] = self._time_range

        async with self._sem:
            async with aiohttp.ClientSession(timeout=self._timeout) as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    response.raise_for_status()
                    data: dict = await response.json()

        results: list[TavilySearchResult] = []
        for item in data.get("results") or []:
            if not isinstance(item, dict):
                continue
            results.append(
                TavilySearchResult(
                    query=query,
                    title=item.get("title"),
                    url=item.get("url"),
                    content=item.get("content"),
                    raw_content=item.get("raw_content"),
                    score=item.get("score"),
                )
            )
        return results


class TavilySmartSearcher:
    """
    SmartSearcher-like research workflow powered by Tavily.
    """

    def __init__(
        self,
        *,
        model: str | GeneralLlm,
        temperature: float | None = None,
        num_searches_to_run: int = 1,
        num_sites_per_search: int = 5,
        search_depth: str = "basic",
        topic: str = "general",
        time_range: str | None = None,
        include_raw_content: bool = False,
        per_result_char_budget: int = 1200,
        total_context_char_budget: int = 12000,
        timeout_seconds: float = 30,
    ) -> None:
        if temperature is not None and not (0 <= temperature <= 1):
            raise ValueError("temperature must be between 0 and 1")
        self._num_searches_to_run = max(1, int(num_searches_to_run))
        self._num_sites_per_search = max(1, int(num_sites_per_search))
        self._per_result_char_budget = max(200, int(per_result_char_budget))
        self._total_context_char_budget = max(1000, int(total_context_char_budget))

        self._llm = (
            model
            if isinstance(model, GeneralLlm)
            else GeneralLlm(model=model, temperature=temperature)
        )
        self._tavily = TavilySearcher(
            timeout_seconds=timeout_seconds,
            search_depth=search_depth,
            topic=topic,
            time_range=time_range,
            include_raw_content=include_raw_content,
        )

    async def invoke(self, prompt: str) -> str:
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("Prompt cannot be empty")

        search_terms = await self._come_up_with_search_queries(prompt)
        results = await self._search(search_terms)
        report = await self._compile_report(results, prompt)
        return self._link_citations(report, results)

    async def _come_up_with_search_queries(self, prompt: str) -> list[str]:
        prompt = clean_indents(
            f"""
            You have been given the following instructions. Instructions are included between <><><><><><><><><><><><> tags.

            <><><><><><><><><><><><>
            {prompt}
            <><><><><><><><><><><><>

            Generate {self._num_searches_to_run} web searches that will help you fulfill any questions in the instructions.
            Make them each target different aspects of the question.
            Please provide the searches as a JSON list of strings like this:
            ["search 1", "search 2"]
            Give no other text than the list of search terms.
            """
        )
        search_terms = await self._llm.invoke_and_return_verified_type(prompt, list[str])
        cleaned = [(term or "").strip() for term in search_terms if (term or "").strip()]
        return cleaned[: self._num_searches_to_run]

    async def _search(self, search_terms: list[str]) -> list[TavilySearchResult]:
        if not search_terms:
            return []

        batches = await asyncio.gather(
            *[
                self._tavily.search(query=term, max_results=self._num_sites_per_search)
                for term in search_terms
            ]
        )
        flattened = [item for batch in batches for item in batch]

        seen: set[str] = set()
        deduped: list[TavilySearchResult] = []
        for item in flattened:
            url = (item.url or "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            deduped.append(item)
        return deduped

    async def _compile_report(
        self, results: list[TavilySearchResult], original_instructions: str
    ) -> str:
        if not results:
            return "No search results found for the query."

        context_lines: list[str] = []
        remaining = self._total_context_char_budget
        for idx, item in enumerate(results, start=1):
            url = item.url or "No URL found"
            title = item.title or "Untitled"
            body = (item.raw_content or item.content or "").strip()
            if not body:
                continue
            if remaining <= 0:
                break
            body = body[: min(len(body), self._per_result_char_budget, remaining)]
            remaining -= len(body)
            context_lines.append(
                f'[{idx}] "{body}". [Source: {url} titled "{title}"]'
            )

        search_result_context = "\n".join(context_lines).strip()
        if not search_result_context:
            return "No usable search results found for the query."

        prompt = clean_indents(
            f"""
            Today is {datetime.now().strftime("%Y-%m-%d")}.
            You have been given the following instructions. Instructions are included between <><><><><><><><><><><><> tags.

            <><><><><><><><><><><><>
            {original_instructions}
            <><><><><><><><><><><><>

            After searching the internet, you found the following results. Results are included between <><><><><><><><><><><><> tags.
            <><><><><><><><><><><><>
            {search_result_context}
            <><><><><><><><><><><><>

            Please follow the instructions and use the search results to answer the question. Unless the instructions specify otherwise, cite your sources inline using [1], [2], etc and use markdown formatting.
            """
        )
        return await self._llm.invoke(prompt)

    @staticmethod
    def _link_citations(text: str, results: list[TavilySearchResult]) -> str:
        urls: list[str] = []
        for item in results:
            url = (item.url or "").strip()
            if not url:
                url = "No URL found"
            urls.append(url)
        return fill_in_citations(urls, text, use_citation_brackets=False)
