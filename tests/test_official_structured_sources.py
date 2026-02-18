import unittest
from types import SimpleNamespace
from unittest.mock import patch

from official_structured_sources import (
    FederalRegisterLimits,
    FredLimits,
    truncate_text,
    prefetch_federal_register,
    prefetch_fred,
    prefetch_bea,
    prefetch_eia,
)


class _FakeResponse:
    def __init__(self, payload: dict, *, url: str = "https://example.test/api") -> None:
        self._payload = payload
        self.url = url

    def raise_for_status(self) -> None:  # noqa: D401 - test stub
        return None

    def json(self) -> dict:
        return self._payload


class TestOfficialStructuredSources(unittest.TestCase):
    def test_truncate_text_marker(self) -> None:
        out = truncate_text("abcdefghij", max_chars=8, marker="...")
        self.assertEqual(out, "abcde...")

    @patch("official_structured_sources.requests.get")
    def test_fred_missing_key_no_network(self, mock_get) -> None:
        mock_get.side_effect = AssertionError("network should not be called")
        out = prefetch_fred(api_key="", search_text="unemployment", limits=FredLimits())
        self.assertEqual(out, "")

    @patch("official_structured_sources.requests.get")
    def test_bea_missing_key_no_network(self, mock_get) -> None:
        mock_get.side_effect = AssertionError("network should not be called")
        q = SimpleNamespace(question_text="GDP in 2026", background_info="")
        out = prefetch_bea(question=q, api_key="", truncation_marker="[TRUNCATED]")
        self.assertEqual(out, "")

    @patch("official_structured_sources.requests.get")
    def test_bea_filters_line_number_and_redacts_key(self, mock_get) -> None:
        payload = {
            "BEAAPI": {
                "Results": {
                    "Data": [
                        {
                            "TableName": "T10105",
                            "LineNumber": "1",
                            "LineDescription": "Gross domestic product",
                            "TimePeriod": "2024Q1",
                            "DataValue": "1,000",
                            "UNIT_MULT": "6",
                            "SeriesCode": "A191RC",
                        },
                        {
                            "TableName": "T10105",
                            "LineNumber": "2",
                            "LineDescription": "Personal consumption expenditures",
                            "TimePeriod": "2024Q1",
                            "DataValue": "500",
                            "UNIT_MULT": "6",
                            "SeriesCode": "DPCERC",
                        },
                        {
                            "TableName": "T10105",
                            "LineNumber": "1",
                            "LineDescription": "Gross domestic product",
                            "TimePeriod": "2024Q2",
                            "DataValue": "1,100",
                            "UNIT_MULT": "6",
                            "SeriesCode": "A191RC",
                        },
                    ]
                }
            }
        }
        mock_get.return_value = _FakeResponse(payload)
        q = SimpleNamespace(question_text="GDP in 2026", background_info="")
        out = prefetch_bea(question=q, api_key="SECRET", truncation_marker="[TRUNCATED]")
        self.assertIn("Gross domestic product", out)
        self.assertIn("2024Q2", out)
        self.assertNotIn("Personal consumption expenditures", out)
        self.assertIn("UserID=REDACTED", out)
        self.assertNotIn("SECRET", out)

    @patch("official_structured_sources.requests.get")
    def test_eia_missing_key_no_network(self, mock_get) -> None:
        mock_get.side_effect = AssertionError("network should not be called")
        q = SimpleNamespace(question_text="WTI crude oil price", background_info="")
        out = prefetch_eia(question=q, api_key="", truncation_marker="[TRUNCATED]")
        self.assertEqual(out, "")

    @patch("official_structured_sources.requests.get")
    def test_federal_register_parsing(self, mock_get) -> None:
        payload = {
            "results": [
                {
                    "title": "Example Rule Title",
                    "publication_date": "2026-02-01",
                    "type": "Rule",
                    "html_url": "https://www.federalregister.gov/documents/2026/02/01/00000-000/example",
                }
            ]
        }
        mock_get.return_value = _FakeResponse(payload, url="https://www.federalregister.gov/api/...")
        out = prefetch_federal_register(
            term="SEC disclosure rule",
            limits=FederalRegisterLimits(max_items=1, max_chars=2000),
            truncation_marker="[TRUNCATED]",
        )
        self.assertIn("Federal Register", out)
        self.assertIn("Example Rule Title", out)
        self.assertIn("https://www.federalregister.gov/documents/2026/02/01/00000-000/example", out)
        self.assertIn("Sources:", out)


if __name__ == "__main__":
    unittest.main()
