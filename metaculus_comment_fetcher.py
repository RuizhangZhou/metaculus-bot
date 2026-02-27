from __future__ import annotations

import logging
import math
import os
import random
import time
from datetime import datetime, timezone

import requests

logger = logging.getLogger(__name__)


_DEFAULT_HTTP_TIMEOUT_SECONDS = 30
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_MIN_DELAY_SECONDS = 0.2
_DEFAULT_BACKOFF_BASE_SECONDS = 1.0


def _parse_iso8601_utc(raw: object) -> datetime | None:
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except Exception:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _try_parse_retry_after_seconds(response: requests.Response) -> float | None:
    raw = response.headers.get("Retry-After")
    if not raw:
        return None
    try:
        value = float(raw.strip())
    except Exception:
        return None
    if value <= 0 or not math.isfinite(value):
        return None
    return value


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip()
    if not raw:
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip()
    if not raw:
        return default
    try:
        return float(raw)
    except Exception:
        return default


def _metaculus_headers(token: str) -> dict[str, str]:
    token = (token or "").strip()
    if not token:
        raise RuntimeError("METACULUS_TOKEN is missing; cannot fetch comments.")
    return {
        "Authorization": f"Token {token}",
        "Accept-Language": "en",
        "User-Agent": "metaculus-bot comment fetcher",
    }


def fetch_my_comments(
    *,
    token: str,
    author_id: int,
    timeout_seconds: int | None = None,
    max_items: int = 50,
    include_private: bool | None = True,
) -> list[dict]:
    """
    Fetches comments authored by `author_id`. Metaculus restricts access: you
    must query your own author ID.

    Returns comment dicts as returned by `/api/comments/`.
    """

    timeout = (
        int(timeout_seconds)
        if isinstance(timeout_seconds, int) and timeout_seconds > 0
        else _env_int(
            "BOT_RETRO_COMMENT_HTTP_TIMEOUT_SECONDS", _DEFAULT_HTTP_TIMEOUT_SECONDS
        )
    )
    max_retries = _env_int("BOT_RETRO_COMMENT_MAX_RETRIES", _DEFAULT_MAX_RETRIES)
    min_delay = _env_float("BOT_RETRO_COMMENT_MIN_DELAY_SECONDS", _DEFAULT_MIN_DELAY_SECONDS)
    backoff_base = _env_float(
        "BOT_RETRO_COMMENT_BACKOFF_BASE_SECONDS", _DEFAULT_BACKOFF_BASE_SECONDS
    )

    session = requests.Session()
    try:
        headers = _metaculus_headers(token)
        collected: list[dict] = []
        offset = 0
        limit = min(100, max(1, int(max_items)))
        while len(collected) < max_items:
            url = "https://www.metaculus.com/api/comments/"
            params: dict[str, object] = {
                "author": author_id,
                "limit": limit,
                "offset": offset,
            }
            if include_private is not None:
                params["is_private"] = "true" if include_private else "false"

            last_exc: Exception | None = None
            resp: requests.Response | None = None
            for attempt in range(max(0, int(max_retries)) + 1):
                if attempt > 0:
                    base = max(0.0, float(backoff_base)) * (2 ** (attempt - 1))
                    jitter = random.uniform(0.0, max(0.1, base * 0.2))
                    time.sleep(base + jitter)

                try:
                    resp = session.get(url, params=params, headers=headers, timeout=timeout)
                except Exception as e:
                    last_exc = e
                    continue

                if resp.status_code == 429:
                    retry_after = _try_parse_retry_after_seconds(resp)
                    sleep_for = (
                        retry_after
                        if retry_after is not None
                        else max(1.0, float(backoff_base)) * (2**attempt)
                    )
                    sleep_for = max(sleep_for, float(min_delay))
                    logger.info(
                        "Metaculus comments rate limited (429); sleeping %.2fs",
                        sleep_for,
                    )
                    time.sleep(sleep_for)
                    last_exc = requests.exceptions.HTTPError(
                        f"429 Too Many Requests for {url}", response=resp
                    )
                    continue

                if 500 <= resp.status_code < 600:
                    last_exc = requests.exceptions.HTTPError(
                        f"{resp.status_code} Server error for {url}", response=resp
                    )
                    continue

                try:
                    resp.raise_for_status()
                    payload = resp.json()
                except Exception as e:
                    last_exc = e
                    continue

                if not isinstance(payload, dict):
                    raise RuntimeError(
                    f"Unexpected comments payload type: {type(payload)}"
                )
                results = payload.get("results")
                if not isinstance(results, list):
                    raise RuntimeError(
                        f"Unexpected comments results type: {type(results)}"
                    )
                for item in results:
                    if isinstance(item, dict):
                        collected.append(item)
                        if len(collected) >= max_items:
                            break

                if min_delay > 0:
                    time.sleep(float(min_delay))

                break
            else:
                raise RuntimeError(
                    "Failed to fetch comments"
                ) from last_exc

            # Stop if fewer than limit returned (no more pages).
            if resp is None:
                break
            try:
                payload = resp.json()
            except Exception:
                break
            results = payload.get("results") if isinstance(payload, dict) else None
            if not isinstance(results, list) or len(results) < limit:
                break
            offset += limit

        return collected[:max_items]
    finally:
        session.close()


def group_comments_by_post_id(comments: list[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = {}
    for item in comments:
        if not isinstance(item, dict):
            continue
        on_post = item.get("on_post")
        if not isinstance(on_post, int):
            continue
        grouped.setdefault(on_post, []).append(item)
    return grouped


def fetch_my_comments_for_post(
    *,
    token: str,
    author_id: int,
    post_id: int,
    timeout_seconds: int | None = None,
    max_items: int = 200,
) -> list[dict]:
    """
    Convenience wrapper that fetches comments by author and filters them locally
    for a specific post_id.

    Note: the Metaculus `/api/comments/` endpoint currently ignores `on_post`
    filters for restricted access, so we must filter client-side.
    """

    comments = fetch_my_comments(
        token=token,
        author_id=author_id,
        timeout_seconds=timeout_seconds,
        max_items=max_items,
    )
    return [c for c in comments if isinstance(c, dict) and c.get("on_post") == post_id]


def select_comment_for_forecast_start_time(
    *, comments: list[dict], forecast_start_time: datetime | None
) -> dict | None:
    if not comments:
        return None

    def comment_created_at(comment: dict) -> datetime:
        dt = _parse_iso8601_utc(comment.get("created_at"))
        return dt or datetime.min.replace(tzinfo=timezone.utc)

    # Sort newest first.
    ordered = sorted(
        [c for c in comments if isinstance(c, dict)],
        key=comment_created_at,
        reverse=True,
    )

    if forecast_start_time is None:
        return ordered[0]

    target = forecast_start_time.astimezone(timezone.utc)
    best: tuple[float, dict] | None = None
    for comment in ordered:
        inc = comment.get("included_forecast")
        if not isinstance(inc, dict):
            continue
        start_dt = _parse_iso8601_utc(inc.get("start_time"))
        if start_dt is None:
            continue
        delta = abs((start_dt - target).total_seconds())
        if best is None or delta < best[0]:
            best = (delta, comment)

    # Accept match within a small tolerance, otherwise return latest comment.
    if best is not None and best[0] <= 10.0:
        return best[1]
    return ordered[0]


def extract_comment_text(comment: dict | None) -> str:
    if not isinstance(comment, dict):
        return ""
    text = comment.get("text")
    return text.strip() if isinstance(text, str) else ""
