from __future__ import annotations

import atexit
import logging
import threading
from dataclasses import dataclass

logger = logging.getLogger(__name__)


def _tavily_credit_cost(search_depth: str) -> int:
    depth = (search_depth or "basic").strip().lower()
    if depth == "advanced":
        return 2
    return 1


@dataclass
class TavilyUsage:
    requests: int = 0
    successes: int = 0
    failures: int = 0
    estimated_credits: int = 0
    raw_content_requests: int = 0
    max_results_total: int = 0
    basic_requests: int = 0
    advanced_requests: int = 0


@dataclass
class ExaFallbackUsage:
    attempts: int = 0
    successes: int = 0
    failures: int = 0


_lock = threading.Lock()
_tavily = TavilyUsage()
_exa_fallback = ExaFallbackUsage()


def reset_search_provider_telemetry() -> None:
    global _tavily, _exa_fallback
    with _lock:
        _tavily = TavilyUsage()
        _exa_fallback = ExaFallbackUsage()


def record_tavily_search_request(
    *,
    search_depth: str,
    max_results: int,
    include_raw_content: bool,
    success: bool,
) -> None:
    depth = (search_depth or "basic").strip().lower()
    with _lock:
        _tavily.requests += 1
        _tavily.estimated_credits += _tavily_credit_cost(depth)
        _tavily.max_results_total += max(0, int(max_results))
        if include_raw_content:
            _tavily.raw_content_requests += 1
        if depth == "advanced":
            _tavily.advanced_requests += 1
        else:
            _tavily.basic_requests += 1
        if success:
            _tavily.successes += 1
        else:
            _tavily.failures += 1


def record_exa_fallback(*, success: bool) -> None:
    with _lock:
        _exa_fallback.attempts += 1
        if success:
            _exa_fallback.successes += 1
        else:
            _exa_fallback.failures += 1


def snapshot_search_provider_telemetry() -> dict[str, dict[str, int]]:
    with _lock:
        return {
            "tavily": {
                "requests": _tavily.requests,
                "successes": _tavily.successes,
                "failures": _tavily.failures,
                "estimated_credits": _tavily.estimated_credits,
                "raw_content_requests": _tavily.raw_content_requests,
                "max_results_total": _tavily.max_results_total,
                "basic_requests": _tavily.basic_requests,
                "advanced_requests": _tavily.advanced_requests,
            },
            "exa_fallback": {
                "attempts": _exa_fallback.attempts,
                "successes": _exa_fallback.successes,
                "failures": _exa_fallback.failures,
            },
        }


def log_search_provider_telemetry_summary() -> None:
    snapshot = snapshot_search_provider_telemetry()
    tavily = snapshot["tavily"]
    exa = snapshot["exa_fallback"]
    if tavily["requests"] == 0 and exa["attempts"] == 0:
        return

    logger.info(
        "Search provider usage summary: "
        "tavily_requests=%s tavily_successes=%s tavily_failures=%s "
        "tavily_estimated_credits=%s tavily_basic_requests=%s "
        "tavily_advanced_requests=%s tavily_raw_content_requests=%s "
        "tavily_max_results_total=%s exa_fallback_attempts=%s "
        "exa_fallback_successes=%s exa_fallback_failures=%s",
        tavily["requests"],
        tavily["successes"],
        tavily["failures"],
        tavily["estimated_credits"],
        tavily["basic_requests"],
        tavily["advanced_requests"],
        tavily["raw_content_requests"],
        tavily["max_results_total"],
        exa["attempts"],
        exa["successes"],
        exa["failures"],
    )


atexit.register(log_search_provider_telemetry_summary)
