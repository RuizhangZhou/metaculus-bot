from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bot.financial_data_context import (
    FinancialDataLimits,
    build_financial_question_spec,
    financial_context_recommends_skipping_broad_news,
    prefetch_financial_data_context,
)


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class TestFinancialDataContext(unittest.TestCase):
    def test_non_financial_question_returns_empty_without_network(self) -> None:
        question = SimpleNamespace(
            question_text="Will a new movie win an award?",
            background_info="",
            resolution_criteria="",
            fine_print="",
        )
        with patch("bot.financial_data_context.requests.get") as mock_get:
            mock_get.side_effect = AssertionError("network should not be called")
            out = prefetch_financial_data_context(question=question)
        self.assertEqual(out, "")

    def test_build_spec_for_index_threshold_question(self) -> None:
        question = SimpleNamespace(
            question_text="Will the S&P 500 close above 7000 on December 31, 2026?",
            background_info="",
            resolution_criteria="Resolved using the official market close.",
            fine_print="",
            scheduled_resolution_time=datetime(2026, 12, 31, tzinfo=timezone.utc),
        )

        spec = build_financial_question_spec(question)

        self.assertIsNotNone(spec)
        assert spec is not None
        self.assertEqual(spec.instruments[0].symbol, "^GSPC")
        self.assertEqual(spec.target_kind, "index_level")
        self.assertEqual(spec.threshold, 7000.0)
        self.assertEqual(spec.threshold_direction, "above")
        self.assertTrue(spec.needs_market_close)
        self.assertTrue(spec.should_skip_broad_news)

    def test_build_spec_from_explicit_yahoo_resolution_url(self) -> None:
        question = SimpleNamespace(
            question_text="Will this index close above 7000?",
            background_info="",
            resolution_criteria=(
                "Resolved using https://finance.yahoo.com/quote/%5EGSPC/ at market close."
            ),
            fine_print="",
        )

        spec = build_financial_question_spec(question)

        self.assertIsNotNone(spec)
        assert spec is not None
        self.assertEqual(spec.instruments[0].symbol, "^GSPC")
        self.assertEqual(spec.instruments[0].source, "explicit-yahoo-url")
        self.assertEqual(spec.target_kind, "index_level")
        self.assertTrue(spec.has_explicit_source_urls)

    def test_build_spec_from_explicit_nasdaq_api_url(self) -> None:
        question = SimpleNamespace(
            question_text="Will this stock close above 250?",
            background_info="",
            resolution_criteria=(
                "Use https://api.nasdaq.com/api/quote/AAPL/info?assetclass=stocks."
            ),
            fine_print="",
        )

        spec = build_financial_question_spec(question)

        self.assertIsNotNone(spec)
        assert spec is not None
        self.assertEqual(spec.instruments[0].symbol, "AAPL")
        self.assertEqual(spec.instruments[0].source, "explicit-nasdaq-api-url")
        self.assertEqual(spec.target_kind, "equity_price")

    @patch("bot.financial_data_context.requests.get")
    def test_prefetch_yahoo_snapshot_renders_quote_and_volatility(self, mock_get) -> None:
        base_ts = 1_767_225_600  # 2026-01-01T00:00:00Z
        timestamps = [base_ts + i * 86_400 for i in range(45)]
        closes = [6000.0 + i * 10.0 for i in range(45)]
        payload = {
            "chart": {
                "result": [
                    {
                        "meta": {
                            "regularMarketPrice": 6450.5,
                            "regularMarketTime": timestamps[-1],
                            "chartPreviousClose": 6440.0,
                            "currency": "USD",
                            "exchangeTimezoneName": "America/New_York",
                            "marketState": "REGULAR",
                        },
                        "timestamp": timestamps,
                        "indicators": {"quote": [{"close": closes}]},
                    }
                ]
            }
        }
        mock_get.return_value = _FakeResponse(payload)
        question = SimpleNamespace(
            question_text="Will the S&P 500 close above 7000?",
            background_info="",
            resolution_criteria="Use the market close.",
            fine_print="",
            scheduled_resolution_time=datetime.now(timezone.utc)
            + timedelta(days=30),
        )

        out = prefetch_financial_data_context(
            question=question,
            limits=FinancialDataLimits(max_symbols=1, max_chars=4000),
        )

        self.assertIn("Financial question spec:", out)
        self.assertIn("S&P 500 (^GSPC)", out)
        self.assertIn("latest_quote: 6450.5 USD", out)
        self.assertIn("trailing_realized_volatility", out)
        self.assertIn("threshold: above 7000", out)
        self.assertIn("mechanical_baseline_probability", out)
        self.assertIn("baseline_probability_for_prompt", out)
        self.assertIn("horizon_days:", out)
        self.assertIn("z_score=", out)
        self.assertIn("query1.finance.yahoo.com/v8/finance/chart/%5EGSPC", out)
        self.assertTrue(financial_context_recommends_skipping_broad_news(out))


if __name__ == "__main__":
    unittest.main()
