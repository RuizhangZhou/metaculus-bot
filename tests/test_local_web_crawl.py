import unittest
from types import SimpleNamespace
from unittest.mock import patch

from local_web_crawl import (
    PlaywrightWebPageParser,
    extract_http_urls,
    normalize_text,
    truncate_text,
)


class TestExtractHttpUrls(unittest.TestCase):
    def test_dedup_and_strip_trailing_punctuation(self) -> None:
        text = (
            "See https://example.com/foo), https://example.com/foo. "
            "and https://example.com/bar] and https://example.com/baz!."
        )
        self.assertEqual(
            extract_http_urls(text),
            [
                "https://example.com/foo",
                "https://example.com/bar",
                "https://example.com/baz",
            ],
        )

    def test_ignores_empty(self) -> None:
        self.assertEqual(extract_http_urls(""), [])


class TestNormalizeText(unittest.TestCase):
    def test_normalizes_whitespace(self) -> None:
        raw = "a\t\tb\r\n\r\n\r\nc\u00a0d  e"
        self.assertEqual(normalize_text(raw), "a b\n\nc d e")


class TestTruncateText(unittest.TestCase):
    def test_truncates_with_marker(self) -> None:
        text = "x" * 20
        out = truncate_text(text, 10, "[...]")
        self.assertEqual(out, "x" * 5 + "[...]")
        self.assertEqual(len(out), 10)

    def test_handles_tiny_max_chars(self) -> None:
        self.assertEqual(truncate_text("hello", 0, "[...]"), "")
        self.assertEqual(truncate_text("hello", 3, "[...]"), "[..")


class TestReadabilityExtraction(unittest.TestCase):
    def test_extract_main_text_prefers_trafilatura_when_available(self) -> None:
        fake_trafilatura = SimpleNamespace(
            extract=lambda html, **kwargs: " Article text from trafilatura. "
        )

        with patch.dict("sys.modules", {"trafilatura": fake_trafilatura}):
            extracted = PlaywrightWebPageParser._extract_main_text(
                "<html><body><article>Ignored fallback.</article></body></html>"
            )

        self.assertEqual(extracted, "Article text from trafilatura.")

    def test_extract_main_text_falls_back_when_trafilatura_returns_empty(self) -> None:
        try:
            import readability  # noqa: F401
            import lxml.html  # noqa: F401
        except Exception:
            self.skipTest("readability-lxml is not installed")

        fake_trafilatura = SimpleNamespace(extract=lambda html, **kwargs: "")
        html = """
        <html>
          <body>
            <nav>Nav</nav>
            <article>
              <h1>Title</h1>
              <p>Hello from readability.</p>
            </article>
          </body>
        </html>
        """

        with patch.dict("sys.modules", {"trafilatura": fake_trafilatura}):
            extracted = PlaywrightWebPageParser._extract_main_text(html)

        self.assertIn("Hello from readability.", extracted)

    def test_extract_main_text_returns_plain_text_when_available(self) -> None:
        try:
            import readability  # noqa: F401
            import lxml.html  # noqa: F401
        except Exception:
            self.skipTest("readability-lxml is not installed")

        html = """
        <html>
          <body>
            <nav>Nav</nav>
            <article>
              <h1>Title</h1>
              <p>Hello world.</p>
            </article>
          </body>
        </html>
        """
        extracted = PlaywrightWebPageParser._extract_main_text(html)
        self.assertIn("Hello world.", extracted)
        self.assertNotIn("<article", extracted)

