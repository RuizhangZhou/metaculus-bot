import asyncio
import logging

logger = logging.getLogger(__name__)


class SmartSearcherCircuitMixin:
    def reset_smart_searcher_circuit_breaker(self) -> None:
        """
        Reset SmartSearcher/Exa circuit breaker state.

        Useful when reusing a single bot instance across multiple tournaments.
        """

        self._smart_searcher_disabled_reason = None
        self._smart_searcher_consecutive_failures = 0

    @staticmethod
    def _is_probably_exa_error(error: BaseException) -> bool:
        error_text = str(error).lower()
        if "api.exa.ai" in error_text or "exa.ai" in error_text:
            return True
        if "exa_api_key" in error_text or "exasearcher" in error_text:
            return True
        if isinstance(error, asyncio.TimeoutError) and "30 seconds" in error_text:
            return True
        return False

    @staticmethod
    def _is_exa_nonrecoverable_error(error: BaseException) -> bool:
        error_text = str(error).lower()
        if "exa_api_key" in error_text and (
            "not set" in error_text or "missing" in error_text
        ):
            return True
        if "invalid api key" in error_text or "unauthorized" in error_text or " 401" in error_text:
            return True
        if "payment required" in error_text or "insufficient credits" in error_text or " 402" in error_text:
            return True
        if "forbidden" in error_text or " 403" in error_text:
            return True
        return False

