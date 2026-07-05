from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse


_OFFICIAL_DOMAINS = {
    "bea.gov",
    "bls.gov",
    "cdc.gov",
    "eia.gov",
    "fda.gov",
    "federalregister.gov",
    "federalreserve.gov",
    "fred.stlouisfed.org",
    "noaa.gov",
    "sec.gov",
    "usgs.gov",
    "who.int",
    "worldbank.org",
    "imf.org",
    "oecd.org",
    "ecb.europa.eu",
    "ecdc.europa.eu",
}

_ACADEMIC_DOMAINS = {
    "arxiv.org",
    "pubmed.ncbi.nlm.nih.gov",
    "ncbi.nlm.nih.gov",
    "nature.com",
    "science.org",
    "thelancet.com",
    "nejm.org",
}

_MARKET_DOMAINS = {
    "kalshi.com",
    "manifold.markets",
    "metaculus.com",
    "polymarket.com",
    "predictit.org",
}

_SOCIAL_OR_FORUM_DOMAINS = {
    "facebook.com",
    "instagram.com",
    "medium.com",
    "reddit.com",
    "substack.com",
    "tiktok.com",
    "twitter.com",
    "x.com",
    "youtube.com",
}


@dataclass(frozen=True)
class SourceQuality:
    domain: str
    source_type: str
    reliability: str
    notes: tuple[str, ...]


def canonical_domain(url: str | None) -> str:
    parsed = urlparse((url or "").strip())
    host = (parsed.hostname or "").strip().lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def _domain_matches(domain: str, candidates: set[str]) -> bool:
    return any(domain == item or domain.endswith(f".{item}") for item in candidates)


def assess_source_quality(
    *,
    url: str | None,
    score: float | None = None,
) -> SourceQuality:
    domain = canonical_domain(url)
    notes: list[str] = []

    if not domain:
        return SourceQuality(
            domain="unknown",
            source_type="unknown",
            reliability="low",
            notes=("missing URL",),
        )

    if domain.endswith(".gov") or domain.endswith(".mil") or _domain_matches(
        domain, _OFFICIAL_DOMAINS
    ):
        source_type = "official_primary"
        reliability = "high"
        notes.append("official or primary-source domain")
    elif domain.endswith(".edu") or _domain_matches(domain, _ACADEMIC_DOMAINS):
        source_type = "academic_research"
        reliability = "high"
        notes.append("academic or research domain")
    elif _domain_matches(domain, _MARKET_DOMAINS):
        source_type = "prediction_market"
        reliability = "medium"
        notes.append("market/community forecast signal")
    elif _domain_matches(domain, _SOCIAL_OR_FORUM_DOMAINS):
        source_type = "social_or_forum"
        reliability = "low"
        notes.append("social, forum, or user-generated source")
    else:
        source_type = "secondary_or_unknown"
        reliability = "medium"
        notes.append("secondary or unclassified source")

    if score is not None:
        try:
            score_value = float(score)
        except (TypeError, ValueError):
            score_value = None
        if score_value is not None and score_value < 0.35:
            notes.append("low search relevance score")

    return SourceQuality(
        domain=domain,
        source_type=source_type,
        reliability=reliability,
        notes=tuple(notes),
    )


def format_source_quality_table(results: list[object], *, max_rows: int = 12) -> str:
    rows: list[str] = [
        "| # | Domain | Type | Reliability | Notes |",
        "|---|---|---|---|---|",
    ]
    row_count = 0
    for idx, item in enumerate(results, start=1):
        if row_count >= max(0, int(max_rows)):
            break
        url = getattr(item, "url", None)
        score = getattr(item, "score", None)
        quality = assess_source_quality(url=url, score=score)
        rows.append(
            "| {idx} | {domain} | {source_type} | {reliability} | {notes} |".format(
                idx=idx,
                domain=_escape_table_cell(quality.domain),
                source_type=_escape_table_cell(quality.source_type),
                reliability=_escape_table_cell(quality.reliability),
                notes=_escape_table_cell("; ".join(quality.notes)),
            )
        )
        row_count += 1

    if row_count == 0:
        return ""
    return "\n".join(rows)


def _escape_table_cell(value: str) -> str:
    return (value or "").replace("|", "\\|").replace("\n", " ").strip()
