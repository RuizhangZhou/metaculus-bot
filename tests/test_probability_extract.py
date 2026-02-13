import unittest

from template_bot_2026 import _extract_probability_percent


class TestExtractProbabilityPercent(unittest.TestCase):
    def test_returns_last_match(self) -> None:
        text = "Draft\nProbability: 10%\n...\nProbability: 42%\n"
        self.assertEqual(_extract_probability_percent(text), 42.0)

    def test_parses_decimals(self) -> None:
        self.assertEqual(_extract_probability_percent("Probability: 4.5%"), 4.5)

    def test_returns_none_when_missing(self) -> None:
        self.assertIsNone(_extract_probability_percent("No forecast here."))

    def test_returns_none_when_out_of_range(self) -> None:
        self.assertIsNone(_extract_probability_percent("Probability: 120%"))

