import html
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlencode

import requests

from forecasting_tools import MetaculusQuestion

logger = logging.getLogger(__name__)


def truncate_text(text: str, *, max_chars: int, marker: str = "\n\n[TRUNCATED]") -> str:
    text = (text or "").strip()
    if max_chars <= 0 or not text:
        return ""
    if len(text) <= max_chars:
        return text
    if len(marker) >= max_chars:
        return marker[:max_chars]
    return text[: max_chars - len(marker)] + marker


def _question_blob(question: MetaculusQuestion) -> str:
    parts = [
        getattr(question, "question_text", None),
        getattr(question, "background_info", None),
        getattr(question, "resolution_criteria", None),
        getattr(question, "fine_print", None),
    ]
    return "\n".join([str(p) for p in parts if isinstance(p, str) and p.strip()]).strip()


def _clean_search_term(text: str, *, max_len: int = 80) -> str:
    text = (text or "").strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^0-9A-Za-z .,&/\\-]", "", text)
    return text[:max_len].strip()


def _ua_headers() -> dict[str, str]:
    ua = os.getenv("BOT_OFFICIAL_HTTP_USER_AGENT", "").strip()
    if not ua:
        ua = "metaculus-bot (official-structured-sources)"
    return {"User-Agent": ua}


@dataclass(frozen=True)
class FederalRegisterLimits:
    timeout_seconds: int = 15
    max_items: int = 5
    days_back: int = 365
    max_chars: int = 4000


def prefetch_federal_register(
    *,
    term: str,
    limits: FederalRegisterLimits | None = None,
    truncation_marker: str = "\n\n[TRUNCATED]",
) -> str:
    term = _clean_search_term(term)
    if not term:
        return ""
    limits = limits or FederalRegisterLimits()

    start_date = (datetime.now(timezone.utc) - timedelta(days=limits.days_back)).date()
    base_url = "https://www.federalregister.gov/api/v1/documents.json"
    params = {
        "per_page": max(1, int(limits.max_items)),
        "order": "newest",
        "conditions[term]": term,
        "conditions[publication_date][gte]": start_date.isoformat(),
    }
    try:
        resp = requests.get(
            base_url,
            params=params,
            timeout=max(1, int(limits.timeout_seconds)),
            headers=_ua_headers(),
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        logger.debug("Federal Register prefetch failed", exc_info=True)
        return ""

    results = data.get("results")
    if not isinstance(results, list) or not results:
        return ""

    lines: list[str] = []
    lines.append(f"Free data sources (Federal Register) search term: {term!r}")
    for item in results[: max(1, int(limits.max_items))]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        pub_date = str(item.get("publication_date") or "").strip()
        doc_type = str(item.get("type") or "").strip()
        html_url = str(item.get("html_url") or "").strip()
        if not html_url:
            continue
        bits = []
        if pub_date:
            bits.append(pub_date)
        if doc_type:
            bits.append(doc_type)
        prefix = f"- {' | '.join(bits)}: " if bits else "- "
        title_part = f"{title} " if title else ""
        lines.append(f"{prefix}{title_part}{html_url}".strip())

    source_url = f"{base_url}?{urlencode(params)}"
    lines.append("Sources:")
    lines.append(f"- {source_url}")
    return truncate_text(
        "\n".join(lines).strip(),
        max_chars=max(1, int(limits.max_chars)),
        marker=truncation_marker,
    )


@dataclass(frozen=True)
class UsgsEarthquakeLimits:
    timeout_seconds: int = 15
    max_items: int = 6
    days_back: int = 7
    min_magnitude: float = 4.5
    max_chars: int = 4000


def prefetch_usgs_earthquakes(
    *,
    limits: UsgsEarthquakeLimits | None = None,
    truncation_marker: str = "\n\n[TRUNCATED]",
) -> str:
    limits = limits or UsgsEarthquakeLimits()
    now = datetime.now(timezone.utc)
    start = (now - timedelta(days=int(limits.days_back))).date().isoformat()
    end = now.date().isoformat()

    base_url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = {
        "format": "geojson",
        "starttime": start,
        "endtime": end,
        "minmagnitude": str(float(limits.min_magnitude)),
        "orderby": "time",
        "limit": str(max(1, int(limits.max_items))),
    }
    try:
        resp = requests.get(
            base_url,
            params=params,
            timeout=max(1, int(limits.timeout_seconds)),
            headers=_ua_headers(),
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        logger.debug("USGS earthquakes prefetch failed", exc_info=True)
        return ""

    features = data.get("features")
    if not isinstance(features, list) or not features:
        return ""

    def fmt_time(ms: Any) -> str:
        try:
            dt = datetime.fromtimestamp(float(ms) / 1000.0, tz=timezone.utc)
        except Exception:
            return ""
        return dt.strftime("%Y-%m-%d %H:%M UTC")

    lines: list[str] = []
    lines.append(
        "Free data sources (USGS earthquakes): "
        f"last {int(limits.days_back)} days, M>={float(limits.min_magnitude):g}"
    )
    for feat in features[: max(1, int(limits.max_items))]:
        if not isinstance(feat, dict):
            continue
        props = feat.get("properties")
        if not isinstance(props, dict):
            continue
        mag = props.get("mag")
        place = str(props.get("place") or "").strip()
        t_ms = props.get("time")
        url = str(props.get("url") or "").strip()
        when = fmt_time(t_ms)
        if not url:
            continue
        mag_part = f"M{float(mag):.1f}" if isinstance(mag, (int, float)) else "M?"
        place_part = f" - {place}" if place else ""
        when_part = f"{when}: " if when else ""
        lines.append(f"- {when_part}{mag_part}{place_part} {url}".strip())

    source_url = f"{base_url}?{urlencode(params)}"
    lines.append("Sources:")
    lines.append(f"- {source_url}")
    return truncate_text(
        "\n".join(lines).strip(),
        max_chars=max(1, int(limits.max_chars)),
        marker=truncation_marker,
    )


@dataclass(frozen=True)
class NoaaNhcLimits:
    timeout_seconds: int = 15
    max_items: int = 3
    max_chars: int = 4000


_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t]+")


def _html_to_text(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return ""
    value = html.unescape(value)
    value = _TAG_RE.sub(" ", value)
    value = value.replace("\r", "\n")
    value = _WS_RE.sub(" ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def _extract_nhc_signal(text: str, *, max_chars: int = 700) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    m = re.search(
        r"(Tropical cyclone formation is not expected during the next 7 days\\.)",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    return truncate_text(text, max_chars=max_chars, marker="…")


def prefetch_noaa_nhc(
    *,
    limits: NoaaNhcLimits | None = None,
    truncation_marker: str = "\n\n[TRUNCATED]",
) -> str:
    limits = limits or NoaaNhcLimits()
    base_url = "https://www.nhc.noaa.gov/gtwo.xml"
    try:
        resp = requests.get(
            base_url,
            timeout=max(1, int(limits.timeout_seconds)),
            headers=_ua_headers(),
        )
        resp.raise_for_status()
        xml_text = resp.text
    except Exception:
        logger.debug("NOAA/NHC prefetch failed", exc_info=True)
        return ""

    try:
        import xml.etree.ElementTree as ET

        root = ET.fromstring(xml_text)
        channel = root.find("channel")
        items = channel.findall("item") if channel is not None else []
    except Exception:
        logger.debug("Failed to parse NOAA/NHC RSS XML", exc_info=True)
        return ""

    if not items:
        return ""

    lines: list[str] = []
    lines.append("Free data sources (NOAA NHC Tropical Weather Outlook):")
    for item in items[: max(1, int(limits.max_items))]:
        title = str(item.findtext("title") or "").strip()
        pub = str(item.findtext("pubDate") or "").strip()
        link = str(item.findtext("link") or "").strip()
        desc = str(item.findtext("description") or "").strip()
        if not link:
            continue
        desc_text = _html_to_text(desc)
        signal = _extract_nhc_signal(desc_text)
        header_bits = " | ".join([b for b in [title, pub] if b])
        if header_bits:
            lines.append(f"- {header_bits}: {link}")
        else:
            lines.append(f"- {link}")
        if signal:
            lines.append(f"  - {signal}")

    lines.append("Sources:")
    lines.append(f"- {base_url}")
    return truncate_text(
        "\n".join(lines).strip(),
        max_chars=max(1, int(limits.max_chars)),
        marker=truncation_marker,
    )


@dataclass(frozen=True)
class FredLimits:
    timeout_seconds: int = 15
    max_series: int = 3
    max_observations: int = 1
    max_chars: int = 4000


def prefetch_fred(
    *,
    api_key: str | None,
    search_text: str,
    limits: FredLimits | None = None,
    truncation_marker: str = "\n\n[TRUNCATED]",
) -> str:
    api_key = (api_key or "").strip()
    if not api_key:
        return ""
    search_text = (search_text or "").strip()
    if not search_text:
        return ""
    limits = limits or FredLimits()

    base_search_url = "https://api.stlouisfed.org/fred/series/search"
    params = {
        "api_key": api_key,
        "file_type": "json",
        "search_text": search_text,
        "order_by": "popularity",
        "sort_order": "desc",
        "limit": str(max(1, int(limits.max_series))),
    }

    try:
        resp = requests.get(
            base_search_url,
            params=params,
            timeout=max(1, int(limits.timeout_seconds)),
            headers=_ua_headers(),
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        logger.debug("FRED series search failed", exc_info=True)
        return ""

    series_list = data.get("seriess")
    if not isinstance(series_list, list) or not series_list:
        return ""

    lines: list[str] = []
    lines.append("Free data sources (FRED series search):")

    obs_limit = max(1, int(limits.max_observations))
    for s in series_list[: max(1, int(limits.max_series))]:
        if not isinstance(s, dict):
            continue
        series_id = str(s.get("id") or "").strip()
        title = str(s.get("title") or "").strip()
        if not series_id:
            continue
        series_url = f"https://fred.stlouisfed.org/series/{series_id}"
        latest_line = ""
        try:
            obs_url = "https://api.stlouisfed.org/fred/series/observations"
            obs_params = {
                "api_key": api_key,
                "file_type": "json",
                "series_id": series_id,
                "sort_order": "desc",
                "limit": str(obs_limit),
            }
            o = requests.get(
                obs_url,
                params=obs_params,
                timeout=max(1, int(limits.timeout_seconds)),
                headers=_ua_headers(),
            )
            o.raise_for_status()
            obs_data = o.json()
            obs = obs_data.get("observations")
            if isinstance(obs, list) and obs:
                last = obs[0] if isinstance(obs[0], dict) else {}
                date = str(last.get("date") or "").strip()
                value = str(last.get("value") or "").strip()
                if date and value:
                    latest_line = f"latest={date}: {value}"
        except Exception:
            latest_line = ""

        header = f"- {title} ({series_id})"
        if latest_line:
            header += f" — {latest_line}"
        lines.append(f"{header}: {series_url}")

    source_params = dict(params)
    source_params["api_key"] = "REDACTED"
    lines.append("Sources:")
    lines.append(f"- {base_search_url}?{urlencode(source_params)}")
    return truncate_text(
        "\n".join(lines).strip(),
        max_chars=max(1, int(limits.max_chars)),
        marker=truncation_marker,
    )


@dataclass(frozen=True)
class BlsLimits:
    timeout_seconds: int = 15
    max_series: int = 4
    max_points: int = 12
    years_back: int = 2
    max_chars: int = 4000


_BLS_SERIES_CATALOG: dict[str, str] = {
    "CUUR0000SA0": "CPI-U (seasonally adjusted), All items, U.S. city average",
    "LNS14000000": "Unemployment rate (seasonally adjusted), percent",
    "CES0000000001": "All employees: Total nonfarm (PAYEMS), thousands",
}


def _select_bls_series_ids(question: MetaculusQuestion, *, max_series: int) -> list[str]:
    text = _question_blob(question).lower()
    chosen: list[str] = []

    def pick(series_id: str) -> None:
        if series_id in chosen:
            return
        if len(chosen) >= max_series:
            return
        chosen.append(series_id)

    if any(k in text for k in ["cpi", "inflation", "consumer price"]):
        pick("CUUR0000SA0")
    if "unemployment" in text or "jobless" in text:
        pick("LNS14000000")
    if any(k in text for k in ["nonfarm payroll", "payroll", "employment", "jobs report"]):
        pick("CES0000000001")

    return chosen


def prefetch_bls(
    *,
    question: MetaculusQuestion,
    registration_key: str | None = None,
    limits: BlsLimits | None = None,
    truncation_marker: str = "\n\n[TRUNCATED]",
) -> str:
    limits = limits or BlsLimits()
    series_ids = _select_bls_series_ids(question, max_series=max(1, int(limits.max_series)))
    if not series_ids:
        return ""

    now = datetime.now(timezone.utc)
    start_year = str(int(now.year) - max(1, int(limits.years_back)))
    end_year = str(int(now.year))
    payload: dict[str, Any] = {
        "seriesid": series_ids,
        "startyear": start_year,
        "endyear": end_year,
    }
    reg_key = (registration_key or "").strip()
    if reg_key:
        payload["registrationKey"] = reg_key

    url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
    try:
        resp = requests.post(
            url,
            json=payload,
            timeout=max(1, int(limits.timeout_seconds)),
            headers=_ua_headers(),
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        logger.debug("BLS prefetch failed", exc_info=True)
        return ""

    results = data.get("Results")
    if not isinstance(results, dict):
        return ""
    series = results.get("series")
    if not isinstance(series, list) or not series:
        return ""

    lines: list[str] = []
    lines.append("Free data sources (BLS time series):")
    max_points = max(1, int(limits.max_points))
    for entry in series[: max(1, int(limits.max_series))]:
        if not isinstance(entry, dict):
            continue
        series_id = str(entry.get("seriesID") or "").strip()
        if not series_id:
            continue
        name = _BLS_SERIES_CATALOG.get(series_id, "")
        header = f"- {series_id}"
        if name:
            header += f" — {name}"
        header += f": https://data.bls.gov/timeseries/{series_id}"
        lines.append(header)

        data_points = entry.get("data")
        if not isinstance(data_points, list):
            continue
        for point in data_points[:max_points]:
            if not isinstance(point, dict):
                continue
            year = str(point.get("year") or "").strip()
            period = str(point.get("period") or "").strip()
            period_name = str(point.get("periodName") or "").strip()
            value = str(point.get("value") or "").strip()
            if not year or not period or not value:
                continue
            # period is like "M01"; keep month name for readability.
            when = f"{year}-{period.lstrip('M')}" if period.startswith("M") else f"{year} {period}"
            label = f"{when} ({period_name})" if period_name else when
            lines.append(f"  - {label}: {value}")

    lines.append("Sources:")
    lines.append(f"- {url}")
    return truncate_text(
        "\n".join(lines).strip(),
        max_chars=max(1, int(limits.max_chars)),
        marker=truncation_marker,
    )


@dataclass(frozen=True)
class BeaLimits:
    timeout_seconds: int = 20
    max_points: int = 8
    years_back: int = 6
    max_chars: int = 4000


@dataclass(frozen=True)
class BeaSeriesSpec:
    table_name: str
    line_number: str
    label: str
    frequency: str = "Q"


_BEA_SPECS: dict[str, BeaSeriesSpec] = {
    "GDP": BeaSeriesSpec(table_name="T10105", line_number="1", label="Gross domestic product"),
    "REAL_GDP": BeaSeriesSpec(
        table_name="T10106", line_number="1", label="Real gross domestic product (chained dollars)"
    ),
    "PCE": BeaSeriesSpec(
        table_name="T20305", line_number="1", label="Personal consumption expenditures"
    ),
}


def _select_bea_specs(question: MetaculusQuestion) -> list[BeaSeriesSpec]:
    text = _question_blob(question).lower()
    specs: list[BeaSeriesSpec] = []
    if "gdp" in text or "gross domestic product" in text:
        specs.append(_BEA_SPECS["GDP"])
    if "real gdp" in text:
        specs.append(_BEA_SPECS["REAL_GDP"])
    if "pce" in text or "personal consumption expenditure" in text:
        specs.append(_BEA_SPECS["PCE"])
    return specs


def prefetch_bea(
    *,
    question: MetaculusQuestion,
    api_key: str | None,
    limits: BeaLimits | None = None,
    truncation_marker: str = "\n\n[TRUNCATED]",
) -> str:
    api_key = (api_key or "").strip()
    if not api_key:
        return ""
    limits = limits or BeaLimits()
    specs = _select_bea_specs(question)
    if not specs:
        return ""

    now = datetime.now(timezone.utc)
    years = [str(y) for y in range(now.year - int(limits.years_back), now.year + 1)]
    year_param = ",".join(years)

    base_url = "https://apps.bea.gov/api/data"

    lines: list[str] = []
    lines.append("Free data sources (BEA API / NIPA):")
    appended_any = False
    for spec in specs:
        params = {
            "UserID": api_key,
            "method": "GetData",
            "datasetname": "NIPA",
            "TableName": spec.table_name,
            "LineNumber": spec.line_number,
            "Year": year_param,
            "Frequency": spec.frequency,
            "ResultFormat": "JSON",
        }
        try:
            resp = requests.get(
                base_url,
                params=params,
                timeout=max(1, int(limits.timeout_seconds)),
                headers=_ua_headers(),
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            logger.debug("BEA fetch failed for %s", spec.label, exc_info=True)
            continue

        api = data.get("BEAAPI")
        if not isinstance(api, dict):
            continue
        results = api.get("Results")
        if not isinstance(results, dict):
            continue
        err = results.get("Error")
        if isinstance(err, dict):
            continue
        rows = results.get("Data")
        if not isinstance(rows, list) or not rows:
            continue

        def parse_value(raw: Any) -> float | None:
            if not isinstance(raw, str):
                return None
            raw = raw.strip()
            if not raw or raw in {"(NA)", "NA"}:
                return None
            raw = raw.replace(",", "")
            try:
                return float(raw)
            except Exception:
                return None

        parsed_rows: list[tuple[str, float]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            period = str(row.get("TimePeriod") or "").strip()
            value = parse_value(row.get("DataValue"))
            if not period or value is None:
                continue
            parsed_rows.append((period, float(value)))
        if not parsed_rows:
            continue
        parsed_rows.sort(key=lambda t: t[0])
        last_rows = parsed_rows[-max(1, int(limits.max_points)) :]

        lines.append(f"- {spec.label} (table {spec.table_name}, line {spec.line_number}):")
        for period, value in last_rows:
            lines.append(f"  - {period}: {value:g}")
        lines.append("  - Source:")
        source_params = dict(params)
        source_params["UserID"] = "REDACTED"
        lines.append(f"    - {base_url}?{urlencode(source_params)}")
        appended_any = True

    if not appended_any:
        return ""

    return truncate_text(
        "\n".join(lines).strip(),
        max_chars=max(1, int(limits.max_chars)),
        marker=truncation_marker,
    )


@dataclass(frozen=True)
class EiaLimits:
    timeout_seconds: int = 20
    max_points: int = 14
    max_chars: int = 4000


_EIA_SERIES_MAP: dict[str, tuple[str, str]] = {
    # EIA series ids are stable; keep this list small + high-signal.
    "WTI_DAILY": ("PET.RWTC.D", "Crude Oil Prices: West Texas Intermediate (WTI) - Cushing, Oklahoma (daily)"),
}


def _select_eia_series(question: MetaculusQuestion) -> list[tuple[str, str]]:
    text = _question_blob(question).lower()
    series: list[tuple[str, str]] = []
    if any(k in text for k in ["wti", "west texas", "crude oil", "oil price", "brent"]):
        series.append(_EIA_SERIES_MAP["WTI_DAILY"])
    return series


def prefetch_eia(
    *,
    question: MetaculusQuestion,
    api_key: str | None,
    limits: EiaLimits | None = None,
    truncation_marker: str = "\n\n[TRUNCATED]",
) -> str:
    api_key = (api_key or "").strip()
    if not api_key:
        return ""
    limits = limits or EiaLimits()
    selected = _select_eia_series(question)
    if not selected:
        return ""

    base_url = "https://api.eia.gov/v2/seriesid/"
    lines: list[str] = []
    lines.append("Free data sources (EIA series):")
    appended_any = False
    max_points = max(1, int(limits.max_points))
    for series_id, label in selected:
        url = f"{base_url}{series_id}"
        params = {"api_key": api_key, "length": max_points}
        try:
            resp = requests.get(
                url,
                params=params,
                timeout=max(1, int(limits.timeout_seconds)),
                headers=_ua_headers(),
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            logger.debug("EIA fetch failed for %s", series_id, exc_info=True)
            continue

        response = data.get("response")
        if not isinstance(response, dict):
            continue
        raw_data = response.get("data")
        if not isinstance(raw_data, list) or not raw_data:
            continue

        lines.append(f"- {label} ({series_id}):")
        for row in raw_data[:max_points]:
            if not isinstance(row, dict):
                continue
            period = str(row.get("period") or "").strip()
            value = row.get("value")
            if not period:
                continue
            if value is None:
                continue
            lines.append(f"  - {period}: {value}")
        lines.append("  - Source:")
        source_params = dict(params)
        source_params["api_key"] = "REDACTED"
        lines.append(f"    - {url}?{urlencode(source_params)}")
        appended_any = True

    if not appended_any:
        return ""

    return truncate_text(
        "\n".join(lines).strip(),
        max_chars=max(1, int(limits.max_chars)),
        marker=truncation_marker,
    )


def derive_official_search_text(question: MetaculusQuestion) -> str:
    title = str(getattr(question, "question_text", "") or "").strip()
    title = re.sub(r"\s+", " ", title)
    return title[:160].strip()
