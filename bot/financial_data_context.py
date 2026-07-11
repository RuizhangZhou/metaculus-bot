from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
import re
import statistics
from typing import Any
from urllib.parse import quote, unquote, urlparse

import requests

from local_web_crawl import extract_http_urls


@dataclass(frozen=True)
class FinancialInstrumentSpec:
    symbol: str
    label: str
    source: str


@dataclass(frozen=True)
class FinancialQuestionSpec:
    instruments: tuple[FinancialInstrumentSpec, ...]
    target_kind: str
    threshold: float | None
    threshold_direction: str | None
    target_datetime: str | None
    needs_market_close: bool
    has_explicit_source_urls: bool
    should_skip_broad_news: bool


@dataclass(frozen=True)
class FinancialDataLimits:
    timeout_seconds: int = 12
    max_symbols: int = 2
    range: str = "6mo"
    interval: str = "1d"
    max_chars: int = 5000
    min_history_points_for_volatility: int = 20
    truncation_marker: str = "\n\n[TRUNCATED]"


@dataclass(frozen=True)
class _YahooPoint:
    date: str
    close: float


@dataclass(frozen=True)
class _YahooSnapshot:
    instrument: FinancialInstrumentSpec
    lines: tuple[str, ...]
    source_url: str
    latest_price: float | None
    latest_daily_close: float | None
    latest_daily_close_date: str | None
    daily_vol_30: float | None
    daily_vol_90: float | None


_SYMBOL_PATTERNS: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (re.compile(r"\b(s&p\s*500|spx|sp\s*500)\b", re.IGNORECASE), "^GSPC", "S&P 500"),
    (
        re.compile(r"\b(nasdaq composite|ixic)\b", re.IGNORECASE),
        "^IXIC",
        "Nasdaq Composite",
    ),
    (re.compile(r"\b(nasdaq[- ]?100|ndx)\b", re.IGNORECASE), "^NDX", "Nasdaq-100"),
    (
        re.compile(r"\b(dow jones|djia|dow industrials?)\b", re.IGNORECASE),
        "^DJI",
        "Dow Jones Industrial Average",
    ),
    (re.compile(r"\brussell\s*2000\b", re.IGNORECASE), "^RUT", "Russell 2000"),
    (re.compile(r"\bvix\b", re.IGNORECASE), "^VIX", "CBOE Volatility Index"),
    (re.compile(r"\b(bitcoin|btc)\b", re.IGNORECASE), "BTC-USD", "Bitcoin USD"),
    (re.compile(r"\b(ethereum|eth)\b", re.IGNORECASE), "ETH-USD", "Ethereum USD"),
    (
        re.compile(r"\b(wti|west texas|crude oil|oil price)\b", re.IGNORECASE),
        "CL=F",
        "WTI crude oil futures",
    ),
    (re.compile(r"\bbrent\b", re.IGNORECASE), "BZ=F", "Brent crude oil futures"),
    (re.compile(r"\bgold\b", re.IGNORECASE), "GC=F", "Gold futures"),
)

_EVENT_OR_FUNDAMENTAL_TERMS = re.compile(
    r"\b("
    r"earnings|eps|revenue|guidance|profit|sales|gdp|cpi|inflation|jobs|payroll|"
    r"unemployment|fed|fomc|rate decision|recession|default|bankruptcy|merger|"
    r"acquisition|approval|lawsuit|tariff|election"
    r")\b",
    re.IGNORECASE,
)

_FINANCIAL_TERMS = re.compile(
    r"\b("
    r"stock|share price|index|close|closing|price|quote|market|trading|"
    r"bitcoin|btc|ethereum|oil|wti|brent|gold|vix|s&p|nasdaq|dow|russell"
    r")\b",
    re.IGNORECASE,
)

_YAHOO_SYMBOL_LABELS = {
    "^GSPC": "S&P 500",
    "^IXIC": "Nasdaq Composite",
    "^NDX": "Nasdaq-100",
    "^DJI": "Dow Jones Industrial Average",
    "^RUT": "Russell 2000",
    "^VIX": "CBOE Volatility Index",
    "BTC-USD": "Bitcoin USD",
    "ETH-USD": "Ethereum USD",
    "CL=F": "WTI crude oil futures",
    "BZ=F": "Brent crude oil futures",
    "GC=F": "Gold futures",
}

_THRESHOLD_RE = re.compile(
    r"\b(?P<direction>above|over|exceed(?:s|ed|ing)?|greater than|at least|"
    r"below|under|less than|at most|no more than|reach(?:es|ed|ing)?|hit(?:s|ting)?)"
    r"[^0-9$%-]{0,45}[$]?(?P<value>-?\d[\d,]*(?:\.\d+)?)\s*(?P<suffix>%|percent)?",
    re.IGNORECASE,
)


def _question_blob(question: Any) -> str:
    parts = [
        getattr(question, "question_text", "") or "",
        getattr(question, "background_info", "") or "",
        getattr(question, "resolution_criteria", "") or "",
        getattr(question, "fine_print", "") or "",
    ]
    return "\n".join(str(p) for p in parts if str(p).strip())


def _add_instrument(
    instruments: list[FinancialInstrumentSpec],
    *,
    symbol: str,
    label: str,
    source: str,
) -> None:
    normalized = symbol.strip().upper()
    if not normalized:
        return
    if any(existing.symbol.upper() == normalized for existing in instruments):
        return
    instruments.append(
        FinancialInstrumentSpec(symbol=symbol.strip(), label=label.strip(), source=source)
    )


def _normalize_market_symbol(symbol: str) -> str:
    symbol = unquote(symbol or "").strip().upper()
    symbol = symbol.strip("/")
    symbol = re.sub(r"[^A-Z0-9.^=-]", "", symbol)
    return symbol[:20]


def _label_for_symbol(symbol: str) -> str:
    normalized = symbol.upper()
    if normalized in _YAHOO_SYMBOL_LABELS:
        return _YAHOO_SYMBOL_LABELS[normalized]
    if normalized.startswith("^"):
        return f"{normalized} index"
    if normalized.endswith("-USD"):
        return normalized.replace("-", " ")
    if normalized.endswith("=F"):
        return f"{normalized} futures"
    return f"{normalized} equity"


def _add_instruments_from_explicit_urls(
    instruments: list[FinancialInstrumentSpec], *, text: str
) -> None:
    for raw_url in extract_http_urls(text):
        parsed = urlparse(raw_url)
        host = parsed.netloc.lower()
        path_parts = [part for part in parsed.path.split("/") if part]
        symbol = ""
        source = ""

        if "finance.yahoo.com" in host:
            if len(path_parts) >= 2 and path_parts[0].lower() == "quote":
                symbol = _normalize_market_symbol(path_parts[1])
                source = "explicit-yahoo-url"
        elif "query1.finance.yahoo.com" in host or "query2.finance.yahoo.com" in host:
            lowered = [part.lower() for part in path_parts]
            if len(path_parts) >= 4 and lowered[:3] == ["v8", "finance", "chart"]:
                symbol = _normalize_market_symbol(path_parts[3])
                source = "explicit-yahoo-chart-url"
        elif "nasdaq.com" in host:
            lowered = [part.lower() for part in path_parts]
            if len(path_parts) >= 3 and lowered[:2] == ["api", "quote"]:
                symbol = _normalize_market_symbol(path_parts[2])
                source = "explicit-nasdaq-api-url"
            elif "quote" in lowered:
                idx = lowered.index("quote")
                if idx + 1 < len(path_parts):
                    symbol = _normalize_market_symbol(path_parts[idx + 1])
                    source = "explicit-nasdaq-url"
            elif "stocks" in lowered:
                idx = lowered.index("stocks")
                if idx + 1 < len(path_parts):
                    symbol = _normalize_market_symbol(path_parts[idx + 1])
                    source = "explicit-nasdaq-url"

        if symbol:
            _add_instrument(
                instruments,
                symbol=symbol,
                label=_label_for_symbol(symbol),
                source=source or "explicit-source-url",
            )


def _extract_threshold(text: str) -> tuple[float | None, str | None]:
    match = _THRESHOLD_RE.search(text or "")
    if not match:
        return None, None
    raw_value = (match.group("value") or "").replace(",", "")
    try:
        value = float(raw_value)
    except ValueError:
        return None, None

    raw_direction = (match.group("direction") or "").lower()
    if any(term in raw_direction for term in ["below", "under", "less", "at most", "no more"]):
        direction = "below"
    elif any(term in raw_direction for term in ["above", "over", "exceed", "greater", "at least"]):
        direction = "above"
    else:
        direction = "reach"
    return value, direction


def _target_datetime(question: Any) -> str | None:
    for attr in ("scheduled_resolution_time", "close_time", "actual_resolution_time"):
        value = getattr(question, attr, None)
        if value is None:
            continue
        if isinstance(value, datetime):
            return value.astimezone(timezone.utc).isoformat()
        return str(value)
    return None


def _target_kind(text: str, instruments: tuple[FinancialInstrumentSpec, ...]) -> str:
    lower = (text or "").lower()
    if re.search(r"\b(eps|earnings per share)\b", lower):
        return "eps"
    if "revenue" in lower or "sales" in lower:
        return "revenue"
    if re.search(r"\b(gdp|cpi|inflation|unemployment|payroll|jobs report)\b", lower):
        return "macro"
    if any(inst.symbol in {"BTC-USD", "ETH-USD"} for inst in instruments):
        return "crypto_price"
    if any(inst.symbol in {"CL=F", "BZ=F", "GC=F"} for inst in instruments):
        return "commodity_price"
    if any(inst.symbol.startswith("^") for inst in instruments):
        return "index_level"
    if instruments:
        return "equity_price"
    return "unknown"


def build_financial_question_spec(
    question: Any, *, inferred_ticker: str | None = None
) -> FinancialQuestionSpec | None:
    text = _question_blob(question)
    if not text:
        return None

    instruments: list[FinancialInstrumentSpec] = []
    _add_instruments_from_explicit_urls(instruments, text=text)
    for pattern, symbol, label in _SYMBOL_PATTERNS:
        if pattern.search(text):
            _add_instrument(
                instruments, symbol=symbol, label=label, source="keyword"
            )

    ticker = (inferred_ticker or "").strip().upper()
    if ticker and _FINANCIAL_TERMS.search(text):
        _add_instrument(
            instruments, symbol=ticker, label=f"{ticker} equity", source="ticker"
        )

    if not instruments:
        return None

    instruments_tuple = tuple(instruments)
    kind = _target_kind(text, instruments_tuple)
    threshold, direction = _extract_threshold(text)
    needs_market_close = bool(
        re.search(r"\b(close|closing|market close|settlement|settle)\b", text, re.IGNORECASE)
    )
    has_urls = bool(extract_http_urls(text))
    event_terms_present = bool(_EVENT_OR_FUNDAMENTAL_TERMS.search(text))
    should_skip_broad_news = bool(
        kind in {"equity_price", "index_level", "crypto_price", "commodity_price"}
        and not event_terms_present
    )

    return FinancialQuestionSpec(
        instruments=instruments_tuple,
        target_kind=kind,
        threshold=threshold,
        threshold_direction=direction,
        target_datetime=_target_datetime(question),
        needs_market_close=needs_market_close,
        has_explicit_source_urls=has_urls,
        should_skip_broad_news=should_skip_broad_news,
    )


def financial_context_recommends_skipping_broad_news(context: str) -> bool:
    return "Broad news default: skip" in (context or "")


def _truncate(text: str, *, max_chars: int, marker: str) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    marker = marker or ""
    if len(marker) >= max_chars:
        return marker[:max_chars]
    return text[: max_chars - len(marker)] + marker


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return f


def _fmt_pct(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value * 100:.2f}%"


def _volatility(points: list[_YahooPoint], *, window: int) -> tuple[float | None, float | None]:
    if len(points) < 2:
        return None, None
    closes = [p.close for p in points if p.close > 0]
    if len(closes) < 2:
        return None, None
    returns = [
        math.log(closes[i] / closes[i - 1])
        for i in range(1, len(closes))
        if closes[i - 1] > 0 and closes[i] > 0
    ]
    returns = returns[-window:]
    if len(returns) < 2:
        return None, None
    daily = statistics.stdev(returns)
    return daily, daily * math.sqrt(252)


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _parse_target_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    raw = value.strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _terminal_threshold_probability(
    *,
    current_price: float,
    threshold: float,
    daily_vol: float,
    days_remaining: float,
    direction: str,
) -> tuple[float, float, float, float]:
    sigma_to_deadline = daily_vol * math.sqrt(days_remaining)
    if sigma_to_deadline <= 0:
        if direction == "below":
            probability = 1.0 if current_price < threshold else 0.0
        else:
            probability = 1.0 if current_price > threshold else 0.0
        distance = threshold - current_price
        return probability, distance, 0.0, math.inf

    if current_price > 0 and threshold > 0:
        distance = math.log(threshold / current_price)
        z_score = distance / sigma_to_deadline
    else:
        absolute_sigma = abs(current_price) * sigma_to_deadline
        if absolute_sigma <= 0:
            absolute_sigma = sigma_to_deadline
        distance = threshold - current_price
        z_score = distance / absolute_sigma

    if direction == "below":
        probability = _normal_cdf(z_score)
    else:
        probability = 1.0 - _normal_cdf(z_score)
    return max(0.0, min(1.0, probability)), threshold - current_price, sigma_to_deadline, z_score


def _hit_threshold_probability(
    *,
    current_price: float,
    threshold: float,
    daily_vol: float,
    days_remaining: float,
) -> tuple[float, float, float, float]:
    probability_terminal, distance, sigma_to_deadline, z_score = (
        _terminal_threshold_probability(
            current_price=current_price,
            threshold=threshold,
            daily_vol=daily_vol,
            days_remaining=days_remaining,
            direction="above" if threshold >= current_price else "below",
        )
    )
    return min(1.0, 2.0 * probability_terminal), distance, sigma_to_deadline, z_score


def _fmt_probability(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value * 100:.1f}%"


def _baseline_probability_lines(
    *, spec: FinancialQuestionSpec, snapshot: _YahooSnapshot
) -> list[str]:
    if spec.threshold is None:
        return []
    target_dt = _parse_target_datetime(spec.target_datetime)
    if target_dt is None:
        return [
            "  - mechanical_baseline_probability: unavailable "
            "(missing parsable target date)"
        ]

    now = datetime.now(timezone.utc)
    days_remaining = (target_dt - now).total_seconds() / 86_400.0
    if days_remaining <= 0:
        return [
            "  - mechanical_baseline_probability: unavailable "
            "(target date is not in the future)"
        ]

    current_price = (
        snapshot.latest_daily_close
        if spec.needs_market_close and snapshot.latest_daily_close is not None
        else snapshot.latest_price
    )
    if current_price is None or current_price <= 0:
        return [
            "  - mechanical_baseline_probability: unavailable "
            "(missing positive current price)"
        ]

    direction = spec.threshold_direction or "above"
    if direction not in {"above", "below", "reach"}:
        direction = "above"
    method = (
        "lognormal terminal threshold approximation"
        if direction in {"above", "below"}
        else "zero-drift barrier-hit approximation"
    )
    probability_rows: list[tuple[str, float, float, float, float]] = []
    for label, daily_vol in (
        ("30d_vol", snapshot.daily_vol_30),
        ("90d_vol", snapshot.daily_vol_90),
    ):
        if daily_vol is None or daily_vol <= 0:
            continue
        if direction == "reach":
            probability, distance, sigma_to_deadline, z_score = _hit_threshold_probability(
                current_price=current_price,
                threshold=spec.threshold,
                daily_vol=daily_vol,
                days_remaining=days_remaining,
            )
        else:
            probability, distance, sigma_to_deadline, z_score = (
                _terminal_threshold_probability(
                    current_price=current_price,
                    threshold=spec.threshold,
                    daily_vol=daily_vol,
                    days_remaining=days_remaining,
                    direction=direction,
                )
            )
        probability_rows.append(
            (label, probability, distance, sigma_to_deadline, z_score)
        )

    if not probability_rows:
        return [
            "  - mechanical_baseline_probability: unavailable "
            "(insufficient realized volatility)"
        ]

    baseline = statistics.fmean(row[1] for row in probability_rows)
    distance = spec.threshold - current_price
    distance_pct = distance / current_price
    lines = [
        "  - mechanical_baseline_probability:",
        f"    - baseline_probability_for_prompt: {_fmt_probability(baseline)}",
        f"    - model: {method}; zero drift, no jumps, no scheduled-news adjustment",
        f"    - current_price_anchor: {current_price:.6g}",
        f"    - threshold: {spec.threshold:.6g}",
        f"    - distance_to_threshold: {distance:.6g} ({distance_pct:.2%} of current)",
        f"    - horizon_days: {days_remaining:.2f}",
    ]
    for label, probability, _, sigma_to_deadline, z_score in probability_rows:
        lines.append(
            f"    - {label}: probability={_fmt_probability(probability)}, "
            f"sigma_to_deadline={sigma_to_deadline:.4f}, z_score={z_score:.3f}"
        )
    lines.append(
        "    - use_as_anchor: adjust only for event risk, source/methodology conflicts, "
        "or known catalysts not reflected in trailing volatility"
    )
    return lines


def _fetch_yahoo_chart(
    instrument: FinancialInstrumentSpec, *, limits: FinancialDataLimits
) -> _YahooSnapshot:
    encoded = quote(instrument.symbol, safe="")
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded}"
        f"?range={quote(limits.range)}&interval={quote(limits.interval)}"
        "&includePrePost=false"
    )
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; MetaculusBot/1.0; "
            "+https://github.com/RuizhangZhou/metaculus-bot)"
        )
    }
    resp = requests.get(url, headers=headers, timeout=max(1, int(limits.timeout_seconds)))
    resp.raise_for_status()
    data = resp.json()
    result = ((data.get("chart") or {}).get("result") or [None])[0]
    if not isinstance(result, dict):
        raise ValueError("Yahoo chart response missing chart.result")

    meta = result.get("meta") if isinstance(result.get("meta"), dict) else {}
    timestamps = result.get("timestamp") if isinstance(result.get("timestamp"), list) else []
    indicators = result.get("indicators") if isinstance(result.get("indicators"), dict) else {}
    quote_data = (indicators.get("quote") or [None])[0]
    closes = quote_data.get("close") if isinstance(quote_data, dict) else []
    points: list[_YahooPoint] = []
    for ts, close in zip(timestamps, closes):
        close_value = _safe_float(close)
        if close_value is None:
            continue
        try:
            date = datetime.fromtimestamp(int(ts), tz=timezone.utc).date().isoformat()
        except Exception:
            continue
        points.append(_YahooPoint(date=date, close=close_value))

    latest_price = _safe_float(meta.get("regularMarketPrice"))
    if latest_price is None and points:
        latest_price = points[-1].close
    previous_close = _safe_float(meta.get("chartPreviousClose"))
    currency = str(meta.get("currency") or "").strip()
    exchange_tz = str(meta.get("exchangeTimezoneName") or "").strip()
    market_state = str(meta.get("marketState") or "").strip()
    quote_time = ""
    regular_market_time = meta.get("regularMarketTime")
    if regular_market_time is not None:
        try:
            quote_time = datetime.fromtimestamp(
                int(regular_market_time), tz=timezone.utc
            ).isoformat()
        except Exception:
            quote_time = ""

    daily30, ann30 = _volatility(points, window=30)
    daily90, ann90 = _volatility(points, window=90)

    lines: list[str] = [
        f"- {instrument.label} ({instrument.symbol}; {instrument.source}):"
    ]
    if latest_price is not None:
        currency_suffix = f" {currency}" if currency else ""
        lines.append(f"  - latest_quote: {latest_price:.6g}{currency_suffix}")
    if quote_time:
        lines.append(f"  - quote_time_utc: {quote_time}")
    if previous_close is not None:
        lines.append(f"  - previous_close: {previous_close:.6g}")
    if points:
        lines.append(f"  - latest_daily_close: {points[-1].date}: {points[-1].close:.6g}")
    if market_state:
        lines.append(f"  - market_state: {market_state}")
    if exchange_tz:
        lines.append(f"  - exchange_timezone: {exchange_tz}")
    if len(points) >= max(2, int(limits.min_history_points_for_volatility)):
        lines.append(
            "  - trailing_realized_volatility: "
            f"30d_daily={_fmt_pct(daily30)}, 30d_annualized={_fmt_pct(ann30)}, "
            f"90d_daily={_fmt_pct(daily90)}, 90d_annualized={_fmt_pct(ann90)}"
        )
    lines.append(f"  - source: {url}")
    latest_daily_close = points[-1].close if points else None
    latest_daily_close_date = points[-1].date if points else None
    return _YahooSnapshot(
        instrument=instrument,
        lines=tuple(lines),
        source_url=url,
        latest_price=latest_price,
        latest_daily_close=latest_daily_close,
        latest_daily_close_date=latest_daily_close_date,
        daily_vol_30=daily30,
        daily_vol_90=daily90,
    )


def prefetch_financial_data_context(
    *,
    question: Any,
    inferred_ticker: str | None = None,
    limits: FinancialDataLimits | None = None,
) -> str:
    limits = limits or FinancialDataLimits()
    spec = build_financial_question_spec(question, inferred_ticker=inferred_ticker)
    if spec is None:
        return ""

    lines: list[str] = []
    lines.append("Financial question spec:")
    lines.append(
        "- instruments: "
        + ", ".join(f"{i.label} ({i.symbol})" for i in spec.instruments)
    )
    lines.append(f"- target_kind: {spec.target_kind}")
    if spec.threshold is not None:
        direction = spec.threshold_direction or "unspecified"
        lines.append(f"- threshold: {direction} {spec.threshold:g}")
    if spec.target_datetime:
        lines.append(f"- target_datetime_utc_or_site: {spec.target_datetime}")
    lines.append(f"- market_close_language: {str(spec.needs_market_close).lower()}")
    lines.append(f"- explicit_source_urls_in_question: {str(spec.has_explicit_source_urls).lower()}")
    broad_news = "skip" if spec.should_skip_broad_news else "allow targeted"
    lines.append(f"- Broad news default: {broad_news}")

    lines.append("")
    lines.append("Financial market data snapshot (fresh; fetched outside cached research):")
    fetched_any = False
    for instrument in spec.instruments[: max(1, int(limits.max_symbols))]:
        try:
            snapshot = _fetch_yahoo_chart(instrument, limits=limits)
        except Exception as exc:
            lines.append(f"- {instrument.label} ({instrument.symbol}): fetch_failed={exc.__class__.__name__}")
            continue
        lines.extend(snapshot.lines)
        if spec.threshold is not None:
            lines.extend(_baseline_probability_lines(spec=spec, snapshot=snapshot))
        fetched_any = True

    if not fetched_any:
        return ""

    if spec.threshold is not None:
        lines.append("")
        lines.append("Threshold interpretation:")
        lines.append(
            "- Compare the latest_quote/latest_daily_close against the threshold, "
            "then scale the distance by trailing daily volatility and time remaining."
        )
    lines.append("")
    lines.append("Forecasting guidance:")
    lines.append(
        "- For market-price, index-level, crypto, or commodity threshold questions, "
        "anchor on the fresh quote/close and trailing volatility before considering narrative news."
    )
    lines.append(
        "- If question-linked or resolution-criteria sources conflict with this snapshot, "
        "prefer the explicit resolution source and cite the conflict."
    )
    return _truncate(
        "\n".join(lines).strip(),
        max_chars=max(1, int(limits.max_chars)),
        marker=limits.truncation_marker,
    )
