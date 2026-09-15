"""Resolve which tournament to forecast on from the API, instead of a constant.

Every Metaculus tournament slug is per-season. Market Pulse rotates quarterly
(``market-pulse-26q3``), and the seasonal bot tournament has been called
``aibq1``, ``fall-aib-2025``, ``spring-aib-2026`` and ``summer-futureeval-2026``
in turn. Hard-coding any of them means the bot silently forecasts nothing the
day the season turns: the run still exits 0, still logs, and just reports
``total_open=0`` forever. That happened here for 149 days.

Pass ``auto`` (or ``auto:<family>``) wherever a slug is expected and the
currently running one is looked up instead.

Deriving the slug from the calendar date does not work. The real windows are::

    25q2  Apr 1  -> Jul 1        26q1  Dec 18 -> Apr 2
    25q3  Jul 2  -> Oct 17       26q2  Mar 30 -> Jul 1
    25q4  Sep 26 -> Jan 7        26q3  Jul 8  -> Oct 1

Consecutive quarters overlap by about three weeks, and a quarter does not start
on the first day of the calendar quarter. So this matches on the API's own
start/close dates, and returns every tournament in the family that is currently
inside its window -- during an overlap that is legitimately two of them.
FutureEval is the exception: it can publish a practice question before its
advertised start, so an API ``is_ongoing`` flag may open that family early, and
``forecasting_end_date`` closes it before the later resolution-only period.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Iterable, Sequence

import requests

logger = logging.getLogger(__name__)

TOURNAMENTS_URL = "https://www.metaculus.com/api/projects/tournaments/"
AUTO_PATTERN = re.compile(r"^auto(?::(?P<family>[a-z0-9-]+))?$", re.IGNORECASE)


@dataclass(frozen=True)
class Family:
    """How to recognise one recurring tournament series in the API listing."""

    name: str
    matches: Callable[[dict], bool]
    # Slugs that are stable by design and may not appear in the public listing.
    constant: tuple[str, ...] = ()
    # FutureEval sometimes opens a practice question before the tournament's
    # advertised start_date.  In that case the API already marks the project as
    # ongoing, and waiting for start_date means missing a real open question.
    allow_prestart_when_ongoing: bool = False
    # FutureEval's close_date includes its resolution period.  Its
    # forecasting_end_date is the useful cutoff for a forecasting bot.
    end_date_fields: tuple[str, ...] = ("close_date",)


def _is_market_pulse(item: dict) -> bool:
    slug = item.get("slug")
    return isinstance(slug, str) and slug.startswith("market-pulse-")


def _is_seasonal_bot_tournament(item: dict) -> bool:
    """The $50k seasonal bot tournament, whatever it is called this season.

    Matched on ``bots_only`` rather than the name: the name has changed every
    season, but a bots-only prize tournament is what it structurally is.
    """
    return (
        item.get("bot_leaderboard_status") == "bots_only"
        and item.get("type") == "tournament"
    )


FAMILIES: dict[str, Family] = {
    "market-pulse": Family(name="market-pulse", matches=_is_market_pulse),
    # MiniBench is unlisted, so it never shows up in the listing. Metaculus
    # documents the slug as permanently "minibench": the currently active round
    # always answers to it, so there is nothing to resolve.
    "minibench": Family(name="minibench", matches=lambda _: False, constant=("minibench",)),
    "futureeval": Family(
        name="futureeval",
        matches=_is_seasonal_bot_tournament,
        allow_prestart_when_ongoing=True,
        end_date_fields=("forecasting_end_date", "close_date"),
    ),
}


def is_auto(value: str) -> bool:
    return bool(AUTO_PATTERN.match((value or "").strip()))


def _parse_dt(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def _first_date(item: dict, fields: tuple[str, ...]) -> datetime | None:
    for field in fields:
        value = _parse_dt(item.get(field))
        if value is not None:
            return value
    return None


def _is_running(item: dict, now: datetime, *, family: Family) -> bool:
    start = _parse_dt(item.get("start_date"))
    end = _first_date(item, family.end_date_fields)
    if end and end <= now:
        return False
    if (
        start
        and start > now
        and not (
            family.allow_prestart_when_ongoing
            and item.get("is_ongoing") is True
        )
    ):
        return False
    return True


def fetch_tournaments(*, token: str | None = None, timeout: int = 30) -> list[dict]:
    headers = {}
    token = token or os.getenv("METACULUS_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Token {token}"
    response = requests.get(TOURNAMENTS_URL, headers=headers, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict):
        payload = payload.get("results", [])
    return [item for item in payload if isinstance(item, dict)]


def resolve_family(
    family_name: str,
    *,
    token: str | None = None,
    now: datetime | None = None,
    listing: Sequence[dict] | None = None,
) -> list[str]:
    """Slugs in ``family_name`` whose forecasting window is active."""
    family = FAMILIES.get(family_name)
    if family is None:
        raise ValueError(
            f"Unknown tournament family {family_name!r}. "
            f"Known: {', '.join(sorted(FAMILIES))}"
        )
    if family.constant:
        return list(family.constant)

    now = now or datetime.now(timezone.utc)
    items = list(listing) if listing is not None else fetch_tournaments(token=token)
    hits = []
    for item in items:
        slug = item.get("slug")
        if not isinstance(slug, str) or not slug:
            continue
        if family.matches(item) and _is_running(item, now, family=family):
            hits.append((_parse_dt(item.get("start_date")) or now, slug))
    return [slug for _, slug in sorted(hits)]


def expand_identifiers(
    identifiers: Iterable[str],
    *,
    default_family: str,
    token: str | None = None,
    now: datetime | None = None,
    listing: Sequence[dict] | None = None,
) -> list[str]:
    """Replace every ``auto`` / ``auto:<family>`` entry with the live slugs.

    Anything that is not an auto marker is passed through untouched, so pinning
    a specific slug still works exactly as before. Order is preserved and
    duplicates are dropped.
    """
    resolved: list[str] = []
    cached_listing = listing
    for raw in identifiers:
        match = AUTO_PATTERN.match((raw or "").strip())
        if not match:
            if raw:
                resolved.append(raw)
            continue
        family = match.group("family") or default_family
        if cached_listing is None and FAMILIES.get(family, Family("", lambda _: False)).constant == ():
            try:
                cached_listing = fetch_tournaments(token=token)
            except Exception as exc:  # noqa: BLE001 - fall through to the error below
                logger.error(f"Could not list tournaments to resolve {raw!r}: {exc!r}")
                raise
        found = resolve_family(family, token=token, now=now, listing=cached_listing)
        if not found:
            logger.warning(
                f"No running tournament found for family {family!r} "
                f"(from {raw!r}); nothing will be forecast for it."
            )
        else:
            logger.info(f"Resolved {raw!r} -> {', '.join(found)}")
        resolved.extend(found)

    seen: set[str] = set()
    unique = []
    for slug in resolved:
        key = str(slug).lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(slug)
    return unique
