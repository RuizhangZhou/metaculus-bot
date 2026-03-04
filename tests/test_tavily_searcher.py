import os
import unittest
from unittest.mock import patch

from bot.tavily_searcher import TavilySearcher


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    async def json(self) -> dict:
        return self._payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class _FakeSession:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response
        self.post_calls: list[tuple[str, dict, dict]] = []

    def post(self, url: str, *, json: dict, headers: dict):  # noqa: A002 - matches aiohttp signature
        self.post_calls.append((url, json, headers))
        return self._response

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class TestTavilySearcher(unittest.IsolatedAsyncioTestCase):
    async def test_requires_api_key(self) -> None:
        previous = os.environ.pop("TAVILY_API_KEY", None)
        self.addCleanup(
            lambda: os.environ.__setitem__("TAVILY_API_KEY", previous)
            if previous is not None
            else os.environ.pop("TAVILY_API_KEY", None)
        )

        with self.assertRaises(ValueError):
            TavilySearcher()

    async def test_search_parses_results_and_sends_auth(self) -> None:
        previous = os.environ.get("TAVILY_API_KEY")
        os.environ["TAVILY_API_KEY"] = "tvly-test"
        self.addCleanup(
            lambda: os.environ.__setitem__("TAVILY_API_KEY", previous)
            if previous is not None
            else os.environ.pop("TAVILY_API_KEY", None)
        )

        fake_payload = {
            "results": [
                {
                    "title": "Result A",
                    "url": "https://example.com/a",
                    "content": "Snippet A",
                    "raw_content": None,
                    "score": 0.9,
                },
                {
                    "title": "Result B",
                    "url": "https://example.com/b",
                    "content": "Snippet B",
                    "raw_content": "Raw B",
                    "score": 0.7,
                },
            ]
        }
        response = _FakeResponse(fake_payload)
        session = _FakeSession(response)

        with patch("bot.tavily_searcher.aiohttp.ClientSession", return_value=session):
            searcher = TavilySearcher(
                timeout_seconds=5,
                search_depth="basic",
                topic="general",
                time_range="week",
                include_raw_content=True,
            )
            results = await searcher.search(query="hello", max_results=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].title, "Result A")
        self.assertEqual(results[0].url, "https://example.com/a")
        self.assertEqual(results[0].content, "Snippet A")

        self.assertEqual(len(session.post_calls), 1)
        url, json_body, headers = session.post_calls[0]
        self.assertEqual(url, "https://api.tavily.com/search")
        self.assertEqual(headers.get("Authorization"), "Bearer tvly-test")
        self.assertEqual(json_body.get("query"), "hello")
        self.assertEqual(json_body.get("max_results"), 2)
        self.assertEqual(json_body.get("include_raw_content"), True)
        self.assertEqual(json_body.get("time_range"), "week")
