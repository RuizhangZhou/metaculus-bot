from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest

from financial_update_logging import (
    FinancialUpdateGatingOptions,
    extract_financial_metrics_from_context,
    filter_questions_for_financial_update_gating,
    question_key,
    record_financial_forecast_results,
    set_cached_financial_context,
)


def _context(*, baseline: float = 0.612, price: float = 6450.5) -> str:
    return f"""
Financial question spec:
- instruments: S&P 500 (^GSPC)
- target_kind: index_level
- threshold: above 7000
- target_datetime_utc_or_site: 2026-12-31T00:00:00+00:00
- market_close_language: true
- explicit_source_urls_in_question: false
- Broad news default: skip

Financial market data snapshot (fresh; fetched outside cached research):
- S&P 500 (^GSPC; keyword):
  - latest_quote: {price} USD
  - quote_time_utc: 2026-07-10T21:28:30+00:00
  - latest_daily_close: 2026-07-10: {price}
  - trailing_realized_volatility: 30d_daily=1.00%, 30d_annualized=15.87%, 90d_daily=1.10%, 90d_annualized=17.46%
  - source: https://query1.finance.yahoo.com/v8/finance/chart/%5EGSPC?range=6mo&interval=1d&includePrePost=false
  - mechanical_baseline_probability:
    - baseline_probability_for_prompt: {baseline * 100:.1f}%
    - model: lognormal terminal threshold approximation; zero drift, no jumps, no scheduled-news adjustment
    - current_price_anchor: {price}
    - threshold: 7000
    - distance_to_threshold: {7000 - price}
    - horizon_days: 30.00
    - 30d_vol: probability={baseline * 100:.1f}%, sigma_to_deadline=0.0548, z_score=0.123
""".strip()


def _question() -> SimpleNamespace:
    return SimpleNamespace(
        id_of_post=1,
        id_of_question=2,
        conditional_type="",
        group_question_option="",
        page_url="https://www.metaculus.com/questions/1/test/",
        question_text="Will the S&P 500 close above 7000?",
        question_type="binary",
        community_prediction_at_access_time=0.60,
        custom_metadata={},
    )


class TestFinancialUpdateLogging(unittest.TestCase):
    def test_extract_financial_metrics_from_context(self) -> None:
        metrics = extract_financial_metrics_from_context(_context())

        self.assertIsNotNone(metrics)
        assert metrics is not None
        self.assertEqual(metrics["symbol"], "^GSPC")
        self.assertEqual(metrics["target_kind"], "index_level")
        self.assertAlmostEqual(metrics["baseline_probability"], 0.612)
        self.assertAlmostEqual(metrics["current_price_anchor"], 6450.5)
        self.assertAlmostEqual(metrics["daily_vol_30"], 0.01)

    def test_gating_skips_when_financial_state_is_unchanged(self) -> None:
        with TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.json"
            log_path = Path(tmp) / "events.jsonl"
            question = _question()
            metrics = extract_financial_metrics_from_context(_context())
            assert metrics is not None
            state_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "questions": {
                            question_key(question): {
                                "last_forecast_at": (
                                    datetime.now(timezone.utc) - timedelta(hours=12)
                                ).isoformat(),
                                "last_forecast_metrics": metrics,
                                "last_prediction": 0.61,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            options = FinancialUpdateGatingOptions(
                enabled=True,
                state_path=state_path,
                log_path=log_path,
                baseline_delta_threshold=0.03,
                price_sigma_threshold=0.50,
                always_forecast_after_hours=72,
                always_forecast_within_days=2,
            )

            kept, counts = filter_questions_for_financial_update_gating(
                questions=[question],
                options=options,
                build_context=lambda _q: _context(),
                tournament_id="market-pulse-26q3",
            )

            self.assertEqual(kept, [])
            self.assertEqual(counts["skipped"], 1)
            self.assertIn("financial_gating_skipped", log_path.read_text(encoding="utf-8"))

    def test_gating_queues_when_baseline_moves(self) -> None:
        with TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.json"
            log_path = Path(tmp) / "events.jsonl"
            question = _question()
            metrics = extract_financial_metrics_from_context(_context(baseline=0.50))
            assert metrics is not None
            state_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "questions": {
                            question_key(question): {
                                "last_forecast_at": (
                                    datetime.now(timezone.utc) - timedelta(hours=12)
                                ).isoformat(),
                                "last_forecast_metrics": metrics,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            options = FinancialUpdateGatingOptions(
                enabled=True,
                state_path=state_path,
                log_path=log_path,
                baseline_delta_threshold=0.03,
                price_sigma_threshold=0.50,
            )

            kept, counts = filter_questions_for_financial_update_gating(
                questions=[question],
                options=options,
                build_context=lambda _q: _context(baseline=0.57),
                tournament_id="market-pulse-26q3",
            )

            self.assertEqual(kept, [question])
            self.assertEqual(counts["queued"], 1)
            self.assertIn("baseline_delta", log_path.read_text(encoding="utf-8"))

    def test_record_financial_forecast_results_updates_state(self) -> None:
        with TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.json"
            log_path = Path(tmp) / "events.jsonl"
            question = _question()
            context = _context()
            metrics = extract_financial_metrics_from_context(context)
            assert metrics is not None
            set_cached_financial_context(question, context, metrics)
            report = SimpleNamespace(question=question, prediction=0.62)
            options = FinancialUpdateGatingOptions(
                enabled=True,
                state_path=state_path,
                log_path=log_path,
            )

            counts = record_financial_forecast_results(
                reports=[report], options=options, tournament_id="market-pulse-26q3"
            )

            self.assertEqual(counts["financial"], 1)
            state = json.loads(state_path.read_text(encoding="utf-8"))
            saved = state["questions"][question_key(question)]
            self.assertEqual(saved["last_prediction"], 0.62)
            self.assertEqual(saved["last_forecast_metrics"]["symbol"], "^GSPC")
            self.assertIn("financial_forecast_recorded", log_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
