import unittest
from types import SimpleNamespace

from bot.source_quality import (
    assess_source_quality,
    canonical_domain,
    format_source_quality_table,
)


class TestSourceQuality(unittest.TestCase):
    def test_canonical_domain_strips_www(self) -> None:
        self.assertEqual(canonical_domain("https://www.cdc.gov/path"), "cdc.gov")

    def test_assesses_official_source_as_high_reliability(self) -> None:
        quality = assess_source_quality(url="https://www.bls.gov/news.release/foo.htm")
        self.assertEqual(quality.source_type, "official_primary")
        self.assertEqual(quality.reliability, "high")

    def test_assesses_prediction_market_as_medium_signal(self) -> None:
        quality = assess_source_quality(url="https://polymarket.com/event/example")
        self.assertEqual(quality.source_type, "prediction_market")
        self.assertEqual(quality.reliability, "medium")

    def test_assesses_social_source_as_low_reliability(self) -> None:
        quality = assess_source_quality(url="https://reddit.com/r/example/post")
        self.assertEqual(quality.source_type, "social_or_forum")
        self.assertEqual(quality.reliability, "low")

    def test_format_source_quality_table_uses_result_indexes(self) -> None:
        table = format_source_quality_table(
            [
                SimpleNamespace(url="https://www.sec.gov/newsroom", score=0.9),
                SimpleNamespace(url="https://medium.com/post", score=0.2),
            ]
        )

        self.assertIn("| 1 | sec.gov | official_primary | high |", table)
        self.assertIn("| 2 | medium.com | social_or_forum | low |", table)
        self.assertIn("low search relevance score", table)
