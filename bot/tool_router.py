import asyncio
from datetime import datetime
import logging
import os
from pathlib import Path

from pydantic import BaseModel

from bot.env import env_bool as _env_bool, env_int as _env_int

from source_catalog import (
    load_catalog as load_source_catalog,
    render_sources_markdown as render_source_catalog_markdown,
    suggest_sources_for_question as suggest_source_catalog_sources,
)

from forecasting_tools import MetaculusQuestion, clean_indents

logger = logging.getLogger(__name__)

_NOTEPAD_TOOL_ROUTER_PLAN_KEY = "tool_router_plan"


class ToolRouterPlan(BaseModel):
    use_web_search: bool
    fetch_sec_filings: bool
    fetch_sec_revenue: bool
    fetch_nasdaq_eps: bool
    fetch_fred: bool
    fetch_bls: bool
    fetch_bea: bool
    fetch_eia: bool
    fetch_federal_register: bool
    fetch_noaa_nhc: bool
    fetch_usgs_earthquakes: bool
    notes: str | None = None


class ToolRouterMixin:
    @staticmethod
    def _tool_router_enabled() -> bool:
        return _env_bool("BOT_ENABLE_TOOL_ROUTER", True)

    @staticmethod
    def _tool_trace_enabled() -> bool:
        return _env_bool("BOT_ENABLE_TOOL_TRACE", True)

    @staticmethod
    def _tool_trace_max_urls() -> int:
        return _env_int("BOT_TOOL_TRACE_MAX_URLS", 25)

    @staticmethod
    def _tool_trace_max_chars() -> int:
        return _env_int("BOT_TOOL_TRACE_MAX_CHARS", 8000)

    @staticmethod
    def _source_catalog_enabled() -> bool:
        return _env_bool("BOT_ENABLE_SOURCE_CATALOG", True)

    @staticmethod
    def _source_catalog_max_items() -> int:
        return _env_int("BOT_SOURCE_CATALOG_MAX_ITEMS", 15)

    @staticmethod
    def _source_catalog_max_chars() -> int:
        return _env_int("BOT_SOURCE_CATALOG_MAX_CHARS", 2500)

    @staticmethod
    def _source_catalog_crawl_max_urls() -> int:
        return _env_int("BOT_SOURCE_CATALOG_CRAWL_MAX_URLS", 2)

    def _source_catalog_query_text(self, question: MetaculusQuestion) -> str:
        parts = [
            getattr(question, "question_text", "") or "",
            getattr(question, "background_info", "") or "",
            getattr(question, "resolution_criteria", "") or "",
            getattr(question, "fine_print", "") or "",
        ]
        return "\n".join(
            [str(p) for p in parts if isinstance(p, str) and p.strip()]
        ).strip()

    def _get_source_catalog_suggestions(
        self, *, question: MetaculusQuestion
    ) -> tuple[str, list[str]]:
        if not self._source_catalog_enabled():
            return "", []
        max_items = max(0, int(self._source_catalog_max_items()))
        if max_items <= 0:
            return "", []

        catalog_path = Path(__file__).resolve().parents[1] / "source_catalog.yaml"
        try:
            text = catalog_path.read_text(encoding="utf-8")
        except Exception:
            return "", []

        catalog = load_source_catalog(text)
        query_text = self._source_catalog_query_text(question)
        suggested = suggest_source_catalog_sources(
            catalog, query_text=query_text, max_items=max_items
        )
        suggested_urls = [
            str(e.get("url")).strip()
            for e in suggested
            if isinstance(e, dict)
            and isinstance(e.get("url"), str)
            and str(e.get("url")).strip()
        ]

        max_chars = max(0, int(self._source_catalog_max_chars()))
        if max_chars <= 0:
            return "", suggested_urls
        rendered_text, rendered_urls = render_source_catalog_markdown(
            suggested, max_chars=max_chars
        )
        return rendered_text, rendered_urls or suggested_urls

    @staticmethod
    def _truncate_for_router(text: str, max_chars: int) -> str:
        text = (text or "").strip()
        if max_chars <= 0 or not text:
            return ""
        if len(text) <= max_chars:
            return text
        marker = "\n\n[TRUNCATED]"
        if len(marker) >= max_chars:
            return marker[:max_chars]
        return text[: max_chars - len(marker)] + marker

    def _default_tool_router_plan(self, *, question: MetaculusQuestion) -> ToolRouterPlan:
        ticker = self._infer_ticker_symbol(question)
        has_ticker = bool(ticker)
        looks_eps = self._looks_like_eps_question(question)
        looks_revenue = self._looks_like_revenue_question(question)
        return ToolRouterPlan(
            use_web_search=_env_bool("BOT_ENABLE_WEB_SEARCH", True),
            fetch_sec_filings=has_ticker
            and _env_bool("BOT_ENABLE_FREE_SEC_FILINGS_PREFETCH", True),
            fetch_sec_revenue=has_ticker
            and looks_revenue
            and _env_bool("BOT_ENABLE_FREE_REVENUE_PREFETCH", True),
            fetch_nasdaq_eps=has_ticker
            and looks_eps
            and _env_bool("BOT_ENABLE_FREE_EPS_PREFETCH", True),
            fetch_fred=False,
            fetch_bls=False,
            fetch_bea=False,
            fetch_eia=False,
            fetch_federal_register=False,
            fetch_noaa_nhc=False,
            fetch_usgs_earthquakes=False,
            notes="fallback-default",
        )

    def _normalize_tool_router_plan(
        self, *, plan: ToolRouterPlan, question: MetaculusQuestion, inferred_ticker: str
    ) -> ToolRouterPlan:
        ticker = (inferred_ticker or "").strip().upper()
        has_ticker = bool(ticker)
        looks_eps = self._looks_like_eps_question(question)
        looks_revenue = self._looks_like_revenue_question(question)

        fetch_sec_filings = bool(
            plan.fetch_sec_filings
            and has_ticker
            and _env_bool("BOT_ENABLE_FREE_SEC_FILINGS_PREFETCH", True)
        )
        fetch_sec_revenue = bool(
            plan.fetch_sec_revenue
            and has_ticker
            and looks_revenue
            and _env_bool("BOT_ENABLE_FREE_REVENUE_PREFETCH", True)
        )
        fetch_nasdaq_eps = bool(
            plan.fetch_nasdaq_eps
            and has_ticker
            and looks_eps
            and _env_bool("BOT_ENABLE_FREE_EPS_PREFETCH", True)
        )

        has_fred_key = bool(os.getenv("FRED_API_KEY", "").strip())
        has_bea_key = bool(os.getenv("BEA_API_KEY", "").strip())
        has_eia_key = bool(os.getenv("EIA_API_KEY", "").strip())

        fetch_fred = bool(
            plan.fetch_fred
            and has_fred_key
            and _env_bool("BOT_ENABLE_FREE_FRED_PREFETCH", True)
        )
        fetch_bls = bool(
            plan.fetch_bls and _env_bool("BOT_ENABLE_FREE_BLS_PREFETCH", True)
        )
        fetch_bea = bool(
            plan.fetch_bea
            and has_bea_key
            and _env_bool("BOT_ENABLE_FREE_BEA_PREFETCH", True)
        )
        fetch_eia = bool(
            plan.fetch_eia
            and has_eia_key
            and _env_bool("BOT_ENABLE_FREE_EIA_PREFETCH", True)
        )
        fetch_federal_register = bool(
            plan.fetch_federal_register
            and _env_bool("BOT_ENABLE_FREE_FEDERAL_REGISTER_PREFETCH", True)
        )
        fetch_noaa_nhc = bool(
            plan.fetch_noaa_nhc
            and _env_bool("BOT_ENABLE_FREE_NOAA_NHC_PREFETCH", True)
        )
        fetch_usgs_earthquakes = bool(
            plan.fetch_usgs_earthquakes
            and _env_bool("BOT_ENABLE_FREE_USGS_EARTHQUAKES_PREFETCH", True)
        )

        use_web_search = bool(
            plan.use_web_search and _env_bool("BOT_ENABLE_WEB_SEARCH", True)
        )

        return ToolRouterPlan(
            use_web_search=use_web_search,
            fetch_sec_filings=fetch_sec_filings,
            fetch_sec_revenue=fetch_sec_revenue,
            fetch_nasdaq_eps=fetch_nasdaq_eps,
            fetch_fred=fetch_fred,
            fetch_bls=fetch_bls,
            fetch_bea=fetch_bea,
            fetch_eia=fetch_eia,
            fetch_federal_register=fetch_federal_register,
            fetch_noaa_nhc=fetch_noaa_nhc,
            fetch_usgs_earthquakes=fetch_usgs_earthquakes,
            notes=(plan.notes or "").strip() or None,
        )

    async def _get_tool_router_plan(
        self, *, question: MetaculusQuestion, local_crawl_context: str
    ) -> ToolRouterPlan:
        inferred_ticker = (self._infer_ticker_symbol(question) or "").strip().upper()
        looks_eps = self._looks_like_eps_question(question)
        looks_revenue = self._looks_like_revenue_question(question)

        llm = self.get_llm("parser", "llm")
        schema = llm.get_schema_format_instructions_for_pydantic_type(ToolRouterPlan)

        max_local_chars = _env_int("BOT_TOOL_ROUTER_LOCAL_CONTEXT_MAX_CHARS", 6000)
        local_excerpt = self._truncate_for_router(local_crawl_context, max_local_chars)

        catalog_text, _ = self._get_source_catalog_suggestions(question=question)
        source_catalog_block = (
            clean_indents(
                f"""
                Reusable source catalog (curated suggestions; may be empty):
                {catalog_text}
                """
            )
            if catalog_text
            else ""
        )

        prompt = clean_indents(
            f"""
            You are a tool-router for a forecasting research assistant.

            Goal: decide which retrieval sources to use for THIS Metaculus question, before writing the research report.
            Cost priority (cheapest/most reliable first):
            1) Local crawl extracts (already available; do not request again).
            2) Free official deterministic sources (SEC/Nasdaq/FRED/BLS/BEA/EIA/Federal Register/NOAA/USGS) when relevant.
            3) Web search (SmartSearcher/Exa or browsing models) only if needed.

            Rules:
            - Prefer to AVOID web search if local extracts + official sources are sufficient to answer the resolution criteria.
            - Only request SEC/Nasdaq tools if a valid US public-company ticker is available.
            - Prefer official sources when the resolution criteria references an agency or official publication
              (e.g., SEC filings, Federal Register rules, NOAA/NHC advisories, USGS feeds).
            - If the question asks for the latest status of an event (e.g., law passed, conflict outcome, election result),
              web search is usually required unless the local extracts already contain up-to-date primary sources.
            - Return ONLY the final JSON object, no other text.

            Output schema:
            {schema}

            Tool meanings:
            - fetch_sec_filings: get recent SEC 10-K/10-Q/8-K links (submissions endpoint).
            - fetch_sec_revenue: get recent quarterly revenue series (companyfacts endpoint).
            - fetch_nasdaq_eps: get analyst consensus EPS forecast (Nasdaq analyst API).
            - fetch_fred: query FRED (macro/finance time series; requires FRED_API_KEY).
            - fetch_bls: query BLS time series (CPI/unemployment/jobs).
            - fetch_bea: query BEA API (GDP/PCE; requires BEA_API_KEY).
            - fetch_eia: query EIA API (energy/oil; requires EIA_API_KEY).
            - fetch_federal_register: search Federal Register documents (rules/notices; no key).
            - fetch_noaa_nhc: fetch NOAA NHC tropical outlook RSS (no key).
            - fetch_usgs_earthquakes: fetch USGS earthquake feed (no key).
            - use_web_search: run SmartSearcher/Exa or a browsing model to find missing info.

            Inputs:
            - Today: {datetime.now().strftime("%Y-%m-%d")}
            - Inferred ticker (may be empty): {inferred_ticker}
            - Heuristics: looks_like_eps={looks_eps}, looks_like_revenue={looks_revenue}

            Question text:
            {question.question_text}

            Background info:
            {question.background_info or ""}

            Resolution criteria:
            {question.resolution_criteria or ""}

            Fine print:
            {question.fine_print or ""}

            Local crawl extracts (truncated):
            {local_excerpt}

            {source_catalog_block}
            """
        )

        plan = await llm.invoke_and_return_verified_type(prompt, ToolRouterPlan)
        return self._normalize_tool_router_plan(
            plan=plan, question=question, inferred_ticker=inferred_ticker
        )

    async def _get_tool_router_plan_cached(
        self, *, question: MetaculusQuestion, local_crawl_context: str
    ) -> ToolRouterPlan:
        if not self._tool_router_enabled():
            return self._default_tool_router_plan(question=question)

        try:
            notepad = await self._get_notepad(question)
        except Exception:
            return await self._get_tool_router_plan(
                question=question, local_crawl_context=local_crawl_context
            )

        existing = notepad.note_entries.get(_NOTEPAD_TOOL_ROUTER_PLAN_KEY)
        if isinstance(existing, ToolRouterPlan):
            return existing
        if isinstance(existing, asyncio.Task):
            try:
                plan = await existing
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.debug(
                    "Tool router cached task failed; falling back", exc_info=True
                )
                plan = self._default_tool_router_plan(question=question)
            notepad.note_entries[_NOTEPAD_TOOL_ROUTER_PLAN_KEY] = plan
            return plan

        task: asyncio.Task[ToolRouterPlan] = asyncio.create_task(
            self._get_tool_router_plan(
                question=question, local_crawl_context=local_crawl_context
            )
        )
        notepad.note_entries[_NOTEPAD_TOOL_ROUTER_PLAN_KEY] = task

        try:
            plan = await task
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug("Tool router plan failed; falling back", exc_info=True)
            plan = self._default_tool_router_plan(question=question)

        notepad.note_entries[_NOTEPAD_TOOL_ROUTER_PLAN_KEY] = plan
        return plan
