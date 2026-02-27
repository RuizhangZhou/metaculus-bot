import argparse
import html
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import requests


METACULUS_POST_API = "https://www.metaculus.com/api/posts/{post_id}/"
SEC_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"
SEC_ARCHIVES_INDEX_URL = (
    "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodashes}/index.json"
)
SEC_ARCHIVES_FILE_URL = (
    "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodashes}/{filename}"
)
NASDAQ_EARNINGS_DATE_URL = "https://api.nasdaq.com/api/analyst/{ticker}/earnings-date"
NASDAQ_EARNINGS_FORECAST_URL = (
    "https://api.nasdaq.com/api/analyst/{ticker}/earnings-forecast"
)


def _parse_utc_datetime(value: str) -> datetime:
    value = value.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _date_to_str(d: datetime) -> str:
    return d.astimezone(timezone.utc).strftime("%Y-%m-%d")


def _env_sec_user_agent() -> str:
    ua = os.getenv("SEC_USER_AGENT", "").strip()
    if ua:
        return ua
    return (
        "metaculus-bot/0.1 (SEC_USER_AGENT not set; "
        "please set SEC_USER_AGENT='your app name (email)')"
    )


def _requests_session(*, user_agent: str) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": user_agent,
            "Accept": "application/json, text/plain, */*",
            "Accept-Encoding": "gzip, deflate",
        }
    )
    return session


def _env_nasdaq_user_agent() -> str:
    ua = os.getenv("NASDAQ_USER_AGENT", "").strip()
    if ua:
        return ua
    return (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    )


def _curl_get_json(url: str, *, headers: dict[str, str], timeout_s: int) -> Any:
    curl = shutil.which("curl") or shutil.which("curl.exe")
    if not curl:
        raise FileNotFoundError("curl not found (needed for Nasdaq API fallback).")

    cmd = [
        curl,
        "-sS",
        "-L",
        "--compressed",
        url,
        "--max-time",
        str(int(timeout_s)),
    ]
    for k, v in headers.items():
        cmd.extend(["-H", f"{k}: {v}"])

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"curl failed ({proc.returncode}): {msg[:200]}")

    return json.loads(proc.stdout)


def _nasdaq_get_json(
    session: requests.Session, url: str, *, timeout_s: int = 30
) -> Any:
    headers = {
        "User-Agent": _env_nasdaq_user_agent(),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nasdaq.com/",
        "Origin": "https://www.nasdaq.com",
    }
    try:
        return _curl_get_json(url, headers=headers, timeout_s=timeout_s)
    except Exception:
        resp = session.get(url, headers=headers, timeout=(min(5, timeout_s), timeout_s))
        resp.raise_for_status()
        return resp.json()


def fetch_nasdaq_consensus_eps(
    session: requests.Session, *, ticker: str
) -> tuple[str | None, float | None, str | None, list[str]]:
    """
    Returns: (announcement, consensus_eps, fiscal_end, source_urls)
    """
    ticker_norm = _normalize_ticker(ticker)
    date_url = NASDAQ_EARNINGS_DATE_URL.format(ticker=ticker_norm)
    forecast_url = NASDAQ_EARNINGS_FORECAST_URL.format(ticker=ticker_norm)
    urls = [date_url, forecast_url]

    date_json: dict[str, Any] | None
    try:
        date_json = _nasdaq_get_json(session, date_url)
    except Exception:
        date_json = None

    forecast_json: dict[str, Any] | None
    try:
        forecast_json = _nasdaq_get_json(session, forecast_url)
    except Exception:
        forecast_json = None

    date_payload = date_json.get("data") if isinstance(date_json, dict) else None
    if not isinstance(date_payload, dict):
        date_payload = {}

    forecast_payload = (
        forecast_json.get("data") if isinstance(forecast_json, dict) else None
    )
    if not isinstance(forecast_payload, dict):
        forecast_payload = {}

    announcement = date_payload.get("announcement")
    report_text = date_payload.get("reportText") or ""

    consensus_eps = None
    fiscal_end = None

    quarterly = forecast_payload.get("quarterlyForecast")
    if isinstance(quarterly, dict):
        rows = quarterly.get("rows")
        if isinstance(rows, list) and rows:
            row0 = rows[0] if isinstance(rows[0], dict) else None
            if isinstance(row0, dict):
                fiscal_end_raw = row0.get("fiscalEnd")
                if fiscal_end_raw:
                    fiscal_end = str(fiscal_end_raw)

                consensus_raw = row0.get("consensusEPSForecast")
                try:
                    consensus_eps = (
                        float(consensus_raw) if consensus_raw is not None else None
                    )
                except Exception:
                    consensus_eps = None

    if (consensus_eps is None or fiscal_end is None) and report_text:
        m = re.search(
            r"fiscal Quarter ending\s+([A-Za-z]{3}\s+\d{4}).*?consensus EPS forecast for the quarter is \$([0-9]+(?:\.[0-9]+)?)",
            report_text,
        )
        if m:
            fiscal_end = fiscal_end or m.group(1)
            if consensus_eps is None:
                try:
                    consensus_eps = float(m.group(2))
                except Exception:
                    consensus_eps = None

    return (
        str(announcement) if announcement else None,
        consensus_eps,
        fiscal_end,
        urls,
    )


def _sec_get_json(session: requests.Session, url: str, *, timeout_s: int = 30) -> Any:
    resp = session.get(url, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()


def _sec_get_text(session: requests.Session, url: str, *, timeout_s: int = 30) -> str:
    resp = session.get(url, timeout=timeout_s)
    resp.raise_for_status()
    return resp.text


@dataclass(frozen=True)
class SecFiling:
    cik_int: int
    accession: str
    accession_nodashes: str
    form: str
    filing_date: str
    report_date: str | None


@dataclass(frozen=True)
class ExtractedEps:
    ticker: str
    eps: float
    sec_url: str
    evidence_snippet: str
    filing: SecFiling


def _normalize_ticker(ticker: str) -> str:
    # Some Metaculus group questions use stylized tickers with confusable unicode
    # characters (e.g. "ΑΜΖΝ" with Greek letters). Normalize those to ASCII.
    confusable_map = str.maketrans(
        {
            # Greek
            "Α": "A",
            "Β": "B",
            "Ε": "E",
            "Ζ": "Z",
            "Η": "H",
            "Ι": "I",
            "Κ": "K",
            "Μ": "M",
            "Ν": "N",
            "Ο": "O",
            "Ρ": "P",
            "Τ": "T",
            "Υ": "Y",
            "Χ": "X",
            "α": "A",
            "β": "B",
            "ε": "E",
            "ζ": "Z",
            "η": "H",
            "ι": "I",
            "κ": "K",
            "μ": "M",
            "ν": "N",
            "ο": "O",
            "ρ": "P",
            "τ": "T",
            "υ": "Y",
            "χ": "X",
            # Cyrillic
            "А": "A",
            "В": "B",
            "Е": "E",
            "К": "K",
            "М": "M",
            "Н": "H",
            "О": "O",
            "Р": "P",
            "С": "C",
            "Т": "T",
            "Х": "X",
            "а": "A",
            "в": "B",
            "е": "E",
            "к": "K",
            "м": "M",
            "н": "H",
            "о": "O",
            "р": "P",
            "с": "C",
            "т": "T",
            "х": "X",
        }
    )
    return re.sub(r"[^A-Z0-9]", "", ticker.translate(confusable_map).strip().upper())


def _ticker_to_cik_int(session: requests.Session, ticker: str) -> int:
    ticker = _normalize_ticker(ticker)
    data = _sec_get_json(session, SEC_TICKER_MAP_URL)
    for entry in data.values():
        if isinstance(entry, dict) and entry.get("ticker") == ticker:
            return int(entry["cik_str"])
    raise KeyError(f"Ticker {ticker!r} not found in SEC ticker map")


def _cik10(cik_int: int) -> str:
    return f"{cik_int:010d}"


def _iter_recent_filings_for_cik(
    session: requests.Session, cik_int: int
) -> list[SecFiling]:
    cik10 = _cik10(cik_int)
    data = _sec_get_json(session, SEC_SUBMISSIONS_URL.format(cik10=cik10))
    recent = data.get("filings", {}).get("recent", {})
    accession_numbers = recent.get("accessionNumber") or []
    forms = recent.get("form") or []
    filing_dates = recent.get("filingDate") or []
    report_dates = recent.get("reportDate") or []
    out: list[SecFiling] = []
    for i in range(len(accession_numbers)):
        accession = str(accession_numbers[i])
        out.append(
            SecFiling(
                cik_int=cik_int,
                accession=accession,
                accession_nodashes=accession.replace("-", ""),
                form=str(forms[i]) if i < len(forms) else "",
                filing_date=str(filing_dates[i]) if i < len(filing_dates) else "",
                report_date=(
                    str(report_dates[i]) if i < len(report_dates) and report_dates[i] else None
                ),
            )
        )
    return out


def _select_candidate_filings(
    filings: list[SecFiling],
    *,
    target_dt_utc: datetime,
    window_days: int,
    forms: tuple[str, ...] = ("8-K", "8-K/A"),
) -> list[SecFiling]:
    start = (target_dt_utc - timedelta(days=window_days)).date()
    end = (target_dt_utc + timedelta(days=window_days)).date()

    candidates = []
    for f in filings:
        if f.form not in forms:
            continue
        try:
            d = datetime.fromisoformat(f.filing_date).date()
        except Exception:
            continue
        if start <= d <= end:
            candidates.append(f)
    # Prefer earliest filing in the window (earnings releases are usually same-day 8-K)
    candidates.sort(key=lambda x: x.filing_date)
    return candidates


def _html_to_text(s: str) -> str:
    s = html.unescape(s)
    s = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", s)
    s = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", s)
    s = re.sub(r"(?is)<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


_EPS_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"(?i)(?:gaap\s+)?diluted\s+earnings\s+per\s+share(?:[^\d]{0,80})?\s+(?:was|were)\s*\$?\s*(-?\d+(?:\.\d+)?)"
    ),
    re.compile(
        r"(?i)(?:gaap\s+)?diluted\s+earnings\s+per\s+share(?:[^\d]{0,80})?\s+of\s*\$?\s*(-?\d+(?:\.\d+)?)"
    ),
    re.compile(
        r"(?i)(?:gaap\s+)?diluted\s+eps(?:[^\d]{0,80})?\s+(?:was|were)\s*\$?\s*(-?\d+(?:\.\d+)?)"
    ),
    re.compile(
        r"(?i)(?:gaap\s+)?diluted\s+eps(?:[^\d]{0,80})?\s+of\s*\$?\s*(-?\d+(?:\.\d+)?)"
    ),
    # Common earnings-release table row format (no "was/were"), e.g. "Diluted earnings per share (EPS) $ 8.88"
    re.compile(
        r"(?i)(?:gaap\s+)?diluted\s+earnings\s+per\s+share(?:\s*\(eps\))?\s*\$\s*(-?\d+(?:\.\d+)?)"
    ),
    re.compile(
        r"(?i)(?:gaap\s+)?diluted\s+eps(?:\s*\(eps\))?\s*\$\s*(-?\d+(?:\.\d+)?)"
    ),
]


def _is_non_gaap_context(text_lower: str, start_idx: int) -> bool:
    # Keep the context window tight: we only want to reject matches where "non-GAAP"
    # actually modifies the EPS phrase, not a previous bullet/table row.
    before = text_lower[max(0, start_idx - 60) : start_idx]
    # Only look a little ahead so we can catch labels like "(non-GAAP)" without
    # accidentally excluding GAAP values followed by "On a non-GAAP basis ...".
    after = text_lower[start_idx : min(len(text_lower), start_idx + 35)]
    window = before + " " + after
    return ("non-gaap" in window) or ("non gaap" in window) or ("adjusted" in window)


def _extract_gaap_diluted_eps_from_text(text: str) -> tuple[float, str] | None:
    text_norm = _html_to_text(text)
    lower = text_norm.lower()

    # Special-case: some companies (e.g. TSLA) include quarterly GAAP diluted EPS only in
    # a multi-column "EPS ... diluted (GAAP)" row. In that case, take the *latest* value
    # in that row segment (usually the last number before YoY/% columns).
    row_key = "eps attributable to common stockholders, diluted (gaap)"
    row_start = lower.find(row_key)
    if row_start >= 0:
        row_end = lower.find(
            "eps attributable to common stockholders, diluted (non-gaap)", row_start
        )
        if row_end < 0:
            row_end = min(len(text_norm), row_start + 3000)
        segment = text_norm[row_start:row_end]
        floats = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", segment)]
        candidates = [x for x in floats if abs(x) <= 25]
        if candidates:
            eps = candidates[-1]
            snippet_start = max(0, row_start - 40)
            snippet_end = min(len(text_norm), row_start + 260)
            snippet = text_norm[snippet_start:snippet_end].strip()
            return eps, snippet

    for pattern in _EPS_PATTERNS:
        for m in pattern.finditer(text_norm):
            if _is_non_gaap_context(lower, m.start()):
                continue
            try:
                eps = float(m.group(1))
            except Exception:
                continue

            # Avoid obvious non-EPS matches like "$7.6 billion" in mixed-metric sentences.
            post = lower[m.end() : min(len(lower), m.end() + 40)]
            if "billion" in post or "million" in post:
                continue

            snippet_start = max(0, m.start() - 120)
            snippet_end = min(len(text_norm), m.end() + 120)
            snippet = text_norm[snippet_start:snippet_end].strip()
            return eps, snippet
    return None


def _find_eps_in_filing(
    session: requests.Session,
    filing: SecFiling,
    *,
    request_delay_s: float,
) -> ExtractedEps | None:
    index_url = SEC_ARCHIVES_INDEX_URL.format(
        cik_int=filing.cik_int, accession_nodashes=filing.accession_nodashes
    )
    index = _sec_get_json(session, index_url)
    items = index.get("directory", {}).get("item", [])
    html_items = []
    for it in items:
        if not isinstance(it, dict):
            continue
        name = str(it.get("name", ""))
        if not (name.endswith(".htm") or name.endswith(".html")):
            continue
        if name.endswith("-index.html") or name.endswith("-index-headers.html"):
            continue
        if name.lower() in {"index.html", "filingsummary.xml"}:
            continue
        size_raw = it.get("size") or 0
        try:
            size = int(size_raw)
        except Exception:
            size = 0
        html_items.append((size, name))

    # Try larger docs first; exhibits/press releases are usually the biggest HTMLs.
    html_items.sort(reverse=True, key=lambda x: x[0])

    for _, filename in html_items[:12]:
        file_url = SEC_ARCHIVES_FILE_URL.format(
            cik_int=filing.cik_int,
            accession_nodashes=filing.accession_nodashes,
            filename=filename,
        )
        time.sleep(request_delay_s)
        content = _sec_get_text(session, file_url)
        extracted = _extract_gaap_diluted_eps_from_text(content)
        if extracted is None:
            continue
        eps, snippet = extracted
        return ExtractedEps(
            ticker="",
            eps=eps,
            sec_url=file_url,
            evidence_snippet=snippet,
            filing=filing,
        )
    return None


def find_gaap_diluted_eps_from_sec(
    session: requests.Session,
    *,
    ticker: str,
    target_dt_utc: datetime,
    window_days: int,
    request_delay_s: float,
) -> ExtractedEps | None:
    ticker_norm = _normalize_ticker(ticker)
    cik_int = _ticker_to_cik_int(session, ticker_norm)
    filings = _iter_recent_filings_for_cik(session, cik_int)
    candidates = _select_candidate_filings(
        filings, target_dt_utc=target_dt_utc, window_days=window_days
    )

    for filing in candidates:
        time.sleep(request_delay_s)
        result = _find_eps_in_filing(
            session, filing, request_delay_s=request_delay_s
        )
        if result is None:
            continue
        return ExtractedEps(
            ticker=ticker_norm,
            eps=result.eps,
            sec_url=result.sec_url,
            evidence_snippet=result.evidence_snippet,
            filing=result.filing,
        )
    return None


def _metaculus_get_post_json(post_id: int) -> dict[str, Any]:
    resp = requests.get(METACULUS_POST_API.format(post_id=post_id), timeout=30)
    resp.raise_for_status()
    data = resp.json()
    assert isinstance(data, dict)
    return data


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch GAAP diluted EPS from SEC filings for each subquestion in a Metaculus group post."
        )
    )
    parser.add_argument("--post-id", type=int, default=41334)
    parser.add_argument(
        "--ticker",
        action="append",
        default=[],
        help="Only process specified ticker(s). Repeatable; commas supported.",
    )
    parser.add_argument(
        "--only-open",
        action="store_true",
        help="Only process subquestions with status='open'. (Also skips SEC scraping.)",
    )
    parser.add_argument(
        "--skip-sec",
        action="store_true",
        help="Skip SEC scraping entirely (faster; useful pre-earnings).",
    )
    parser.add_argument("--window-days", type=int, default=7)
    parser.add_argument("--sec-delay-seconds", type=float, default=0.15)
    args = parser.parse_args(argv)

    user_agent = _env_sec_user_agent()
    session = _requests_session(user_agent=user_agent)

    tickers_filter: set[str] = set()
    for raw in args.ticker:
        for part in str(raw).split(","):
            t = _normalize_ticker(part)
            if t:
                tickers_filter.add(t)

    post = _metaculus_get_post_json(args.post_id)
    group = post.get("group_of_questions")
    if not isinstance(group, dict):
        raise SystemExit("Post is not a group question (missing group_of_questions).")

    title = post.get("title") or f"post {args.post_id}"
    print(f"# {title} (post_id={args.post_id})")
    print("")

    questions = group.get("questions") or []
    if not isinstance(questions, list):
        raise SystemExit("Unexpected API shape: group_of_questions.questions is not a list.")

    rows: list[dict[str, Any]] = []
    for q in questions:
        if not isinstance(q, dict):
            continue
        ticker = str(q.get("label") or "").strip()
        if not ticker:
            continue
        ticker_norm = _normalize_ticker(ticker)
        if tickers_filter and ticker_norm not in tickers_filter:
            continue

        scheduled_close_time = q.get("scheduled_close_time")
        if not isinstance(scheduled_close_time, str) or not scheduled_close_time:
            continue
        target_dt_utc = _parse_utc_datetime(scheduled_close_time)
        status = str(q.get("status") or "")
        resolution = q.get("resolution")

        if args.only_open and status.lower() != "open":
            continue

        extracted = None
        if not (args.skip_sec or status.lower() == "open" or args.only_open):
            extracted = find_gaap_diluted_eps_from_sec(
                session,
                ticker=ticker,
                target_dt_utc=target_dt_utc,
                window_days=int(args.window_days),
                request_delay_s=float(args.sec_delay_seconds),
            )
        nasdaq_announcement, nasdaq_consensus, nasdaq_fiscal_end, nasdaq_url = (
            fetch_nasdaq_consensus_eps(session, ticker=ticker)
        )

        rows.append(
            {
                "ticker": ticker,
                "status": status,
                "scheduled_close": _date_to_str(target_dt_utc),
                "metaculus_resolution": resolution,
                "sec_eps": extracted.eps if extracted else None,
                "sec_url": extracted.sec_url if extracted else None,
                "evidence": extracted.evidence_snippet if extracted else None,
                "nasdaq_announcement": nasdaq_announcement,
                "nasdaq_consensus": nasdaq_consensus,
                "nasdaq_fiscal_end": nasdaq_fiscal_end,
                "nasdaq_urls": nasdaq_url,
            }
        )

    rows.sort(key=lambda r: str(r["ticker"]))
    for r in rows:
        ticker = r["ticker"]
        status = r["status"]
        close = r["scheduled_close"]
        met = r["metaculus_resolution"]
        sec_eps = r["sec_eps"]
        print(f"- {ticker} ({status}, close={close})")
        print(f"  - Metaculus resolution: {met}")
        if sec_eps is None:
            if args.skip_sec or status.lower() == "open":
                print("  - SEC: skipped")
            else:
                print("  - SEC: not found (yet) in window")
        else:
            print(f"  - SEC extracted GAAP diluted EPS: {sec_eps}")
            print(f"  - SEC source: {r['sec_url']}")
            print(f"  - Evidence: {r['evidence']}")
        if r.get("nasdaq_consensus") is not None or r.get("nasdaq_announcement"):
            detail = []
            if r.get("nasdaq_announcement"):
                detail.append(str(r["nasdaq_announcement"]))
            if r.get("nasdaq_consensus") is not None:
                fe = r.get("nasdaq_fiscal_end") or "unknown fiscal end"
                detail.append(f"consensus EPS {r['nasdaq_consensus']} (fiscal end {fe})")
            print(f"  - Nasdaq: {'; '.join(detail)}")
            urls = r.get("nasdaq_urls") or []
            if isinstance(urls, list):
                for u in urls:
                    print(f"  - Nasdaq source: {u}")
            else:
                print(f"  - Nasdaq source: {urls}")
        print("")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
