import unittest

from bot.search_telemetry import (
    record_exa_fallback,
    record_tavily_search_request,
    reset_search_provider_telemetry,
    snapshot_search_provider_telemetry,
)


class TestSearchTelemetry(unittest.TestCase):
    def setUp(self) -> None:
        reset_search_provider_telemetry()

    def tearDown(self) -> None:
        reset_search_provider_telemetry()

    def test_tavily_credit_estimate_uses_search_depth(self) -> None:
        record_tavily_search_request(
            search_depth="basic",
            max_results=8,
            include_raw_content=False,
            success=True,
        )
        record_tavily_search_request(
            search_depth="advanced",
            max_results=5,
            include_raw_content=True,
            success=False,
        )

        telemetry = snapshot_search_provider_telemetry()
        self.assertEqual(telemetry["tavily"]["requests"], 2)
        self.assertEqual(telemetry["tavily"]["successes"], 1)
        self.assertEqual(telemetry["tavily"]["failures"], 1)
        self.assertEqual(telemetry["tavily"]["estimated_credits"], 3)
        self.assertEqual(telemetry["tavily"]["max_results_total"], 13)
        self.assertEqual(telemetry["tavily"]["raw_content_requests"], 1)
        self.assertEqual(telemetry["tavily"]["basic_requests"], 1)
        self.assertEqual(telemetry["tavily"]["advanced_requests"], 1)

    def test_exa_fallback_counts_successes_and_failures(self) -> None:
        record_exa_fallback(success=True)
        record_exa_fallback(success=False)

        telemetry = snapshot_search_provider_telemetry()
        self.assertEqual(telemetry["exa_fallback"]["attempts"], 2)
        self.assertEqual(telemetry["exa_fallback"]["successes"], 1)
        self.assertEqual(telemetry["exa_fallback"]["failures"], 1)
