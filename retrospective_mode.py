from __future__ import annotations

import logging
import math
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests

from digest_mode import (
    _extract_account_prediction_from_question_json,
    _get_numeric_percentile_map,
    _latest_forecast_entry,
)
from forecasting_tools import MetaculusClient, MetaculusQuestion
from forecasting_tools.helpers.metaculus_client import ApiFilter
from tournament_update import _extract_cp_at_time, _extract_cp_latest

logger = logging.getLogger(__name__)


_DEFAULT_MAX_QUESTIONS = 100
_DEFAULT_HTTP_TIMEOUT_SECONDS = 30
_DEFAULT_API2_MAX_RETRIES = 4
_DEFAULT_API2_MIN_DELAY_SECONDS = 0.2
_DEFAULT_API2_BACKOFF_BASE_SECONDS = 1.0


@dataclass(frozen=True)
class RetrospectiveRow:
    question_id: int
    post_id: int
    url: str
    title: str
    question_type: str
    resolved_at: datetime | None
    resolution: object | None
    my_forecast_at: datetime | None
    my_prediction: object | None
    community_at_my_forecast: object | None
    community_latest: object | None
    score: float | None
    notes: str


def _parse_iso8601_utc(raw: object) -> datetime | None:
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text:
        return None
    # Metaculus uses a 'Z' suffix for UTC.
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except Exception:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _timestamp_to_dt(raw: object) -> datetime | None:
    if not isinstance(raw, (int, float)):
        return None
    ts = float(raw)
    if not math.isfinite(ts):
        return None
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    except Exception:
        return None


def _metaculus_api2_headers() -> dict[str, str]:
    token = os.getenv("METACULUS_TOKEN", "").strip()
    if not token:
        raise RuntimeError("METACULUS_TOKEN is missing; cannot query api2.")
    return {
        "Authorization": f"Token {token}",
        "User-Agent": "metac-bot-template retrospective (https://github.com/Metaculus/metac-bot-template)",
    }


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


def _fetch_api2_post(
    *,
    session: requests.Session,
    post_id: int,
    timeout_seconds: int,
    max_retries: int,
    min_delay_seconds: float,
    backoff_base_seconds: float,
) -> dict:
    url = f"https://www.metaculus.com/api2/questions/{post_id}/?include=aggregations"
    headers = _metaculus_api2_headers()

    last_exc: Exception | None = None
    for attempt in range(max(0, int(max_retries)) + 1):
        if attempt > 0:
            base = max(0.0, float(backoff_base_seconds)) * (2**(attempt - 1))
            jitter = random.uniform(0.0, max(0.1, base * 0.2))
            time.sleep(base + jitter)

        try:
            resp = session.get(url, headers=headers, timeout=timeout_seconds)
        except Exception as e:
            last_exc = e
            continue

        if resp.status_code == 429:
            retry_after = _try_parse_retry_after_seconds(resp)
            sleep_for = (
                retry_after
                if retry_after is not None
                else max(1.0, float(backoff_base_seconds)) * (2**attempt)
            )
            sleep_for = max(sleep_for, float(min_delay_seconds))
            logger.info(
                "Metaculus api2 rate limited (429) for post_id=%s; sleeping %.2fs",
                post_id,
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

        if min_delay_seconds > 0:
            time.sleep(float(min_delay_seconds))

        break
    else:
        raise RuntimeError(f"Failed to fetch api2 payload for post_id={post_id}") from last_exc

    if not isinstance(payload, dict):
        raise RuntimeError(
            f"Unexpected api2 response for post_id={post_id}: {type(payload)}"
        )
    return payload


def _select_api2_question_json(*, api2_post: dict, question_id: int) -> dict | None:
    # Normal post: {"question": {...}}
    question = api2_post.get("question")
    if isinstance(question, dict):
        return question

    # Group post: {"group_of_questions": {"questions": [{...}, ...]}}
    group = api2_post.get("group_of_questions")
    if not isinstance(group, dict):
        return None
    questions = group.get("questions")
    if not isinstance(questions, list):
        return None
    for item in questions:
        if isinstance(item, dict) and item.get("id") == question_id:
            return item
    return None


def _question_type_from_json(question_json: dict) -> str:
    qtype = question_json.get("type")
    if isinstance(qtype, str) and qtype:
        return qtype
    return "unknown"


def _resolution_to_binary_outcome(resolution: object) -> float | None:
    if isinstance(resolution, bool):
        return 1.0 if resolution else 0.0
    if isinstance(resolution, (int, float)):
        if float(resolution) in (0.0, 1.0):
            return float(resolution)
        return None
    if isinstance(resolution, str):
        lowered = resolution.strip().lower()
        if lowered in {"yes", "true", "1"}:
            return 1.0
        if lowered in {"no", "false", "0"}:
            return 0.0
    return None


def _score_binary(*, prediction: object, outcome: float | None) -> float | None:
    if outcome is None:
        return None
    if not isinstance(prediction, (int, float)):
        return None
    p = float(prediction)
    if not (0.0 <= p <= 1.0):
        return None
    return (p - float(outcome)) ** 2


def _parse_float_maybe(raw: object) -> float | None:
    if isinstance(raw, (int, float)):
        value = float(raw)
        if math.isfinite(value):
            return value
        return None
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        try:
            value = float(text)
        except Exception:
            return None
        return value if math.isfinite(value) else None
    return None


def _extract_last_forecast_time(question_json: dict) -> datetime | None:
    latest = _latest_forecast_entry(question_json.get("my_forecasts"))
    if not isinstance(latest, dict):
        return None
    return _timestamp_to_dt(latest.get("start_time"))


def _format_pred(pred: object | None, *, question_type: str) -> str:
    if pred is None:
        return "n/a"
    if question_type == "binary" and isinstance(pred, (int, float)):
        return f"{float(pred):.3f}"
    if question_type in {"numeric", "date", "discrete"} and isinstance(pred, dict):
        pmap = _get_numeric_percentile_map(pred)
        if pmap:
            # Pick nearest available percentiles when exact values are missing.
            def approx(target: float) -> float | None:
                if not pmap:
                    return None
                nearest = min(pmap.keys(), key=lambda p: abs(p - target))
                return pmap.get(nearest)

            p10 = approx(0.1)
            p50 = approx(0.5)
            p90 = approx(0.9)
            if question_type == "date":
                def fmt_ts(ts: float | None) -> str:
                    if ts is None:
                        return "?"
                    try:
                        return datetime.fromtimestamp(
                            float(ts), tz=timezone.utc
                        ).date().isoformat()
                    except Exception:
                        return str(ts)

                return f"p50~={fmt_ts(p50)} (p10~={fmt_ts(p10)}, p90~={fmt_ts(p90)})"

            def fmt_num(x: float | None) -> str:
                if x is None:
                    return "?"
                return f"{float(x):.4g}"

            return f"p50~={fmt_num(p50)} (p10~={fmt_num(p10)}, p90~={fmt_num(p90)})"
    if isinstance(pred, dict):
        # Show top-3 keys by probability if it looks like a probability map.
        items: list[tuple[str, float]] = []
        for k, v in pred.items():
            if isinstance(k, str) and isinstance(v, (int, float)):
                items.append((k, float(v)))
        if items:
            items.sort(key=lambda kv: kv[1], reverse=True)
            top = ", ".join(f"{k}={v:.2f}" for k, v in items[:3])
            return "{" + top + ("..." if len(items) > 3 else "") + "}"
        return "{...}"
    if isinstance(pred, (int, float)):
        return f"{float(pred):.4g}"
    return str(pred)


def _safe_title(question: MetaculusQuestion) -> str:
    text = getattr(question, "question_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    return f"question:{getattr(question, 'id_of_question', '')}"


def _safe_url(question: MetaculusQuestion) -> str:
    url = getattr(question, "page_url", None)
    if isinstance(url, str) and url.strip():
        return url.strip()
    post_id = getattr(question, "id_of_post", None)
    if isinstance(post_id, int):
        return f"https://www.metaculus.com/questions/{post_id}"
    return "unknown"


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


async def _get_resolved_questions_i_forecasted(
    *, client: MetaculusClient, tournament_id: str, max_questions: int
) -> list[MetaculusQuestion]:
    api_filter = ApiFilter(
        allowed_tournaments=[tournament_id],
        allowed_statuses=["resolved"],
        is_previously_forecasted_by_user=True,
        group_question_mode="unpack_subquestions",
    )
    # The Metaculus API returns up to 100 posts per request. If we're only
    # looking at the first page (typical for tournaments), avoid pagination
    # to reduce rate-limit risk.
    num_questions = max_questions if max_questions > 100 else None
    questions = await client.get_questions_matching_filter(
        api_filter,
        num_questions=num_questions,
        randomly_sample=False,
        error_if_question_target_missed=False,
    )
    return questions[:max_questions] if max_questions > 0 else questions


def _render_markdown(*, tournament_id: str, rows: list[RetrospectiveRow]) -> str:
    now = datetime.now(timezone.utc).isoformat()
    lines: list[str] = []
    lines.append(f"# Retrospective: {tournament_id}")
    lines.append("")
    lines.append(f"Generated: {now}")
    lines.append("")

    if not rows:
        lines.append("No resolved questions with bot forecasts found.")
        lines.append("")
        return "\n".join(lines)

    scored = [r for r in rows if isinstance(r.score, (int, float))]
    binary_scored = [r for r in scored if r.question_type == "binary"]
    if binary_scored:
        avg_brier = sum(float(r.score) for r in binary_scored) / len(binary_scored)
        lines.append(f"- Binary: {len(binary_scored)} scored, avg Brier={avg_brier:.4f}")
    if scored and len(scored) != len(binary_scored):
        lines.append(f"- Total scored: {len(scored)} (non-binary scoring is partial)")
    lines.append("")

    worst = sorted(
        (r for r in binary_scored if r.score is not None),
        key=lambda r: float(r.score),
        reverse=True,
    )[:10]
    if worst:
        lines.append("## Biggest binary misses (Brier)")
        lines.append("")
        for r in worst:
            lines.append(
                f"- {r.score:.3f} | my={_format_pred(r.my_prediction, question_type=r.question_type)} "
                f"cp_at_forecast={_format_pred(r.community_at_my_forecast, question_type=r.question_type)} "
                f"cp_latest={_format_pred(r.community_latest, question_type=r.question_type)} "
                f"| {r.url}"
            )
        lines.append("")

    lines.append("## Details")
    lines.append("")
    for r in rows:
        lines.append(f"### {r.title}")
        lines.append(f"- URL: {r.url}")
        lines.append(f"- Type: {r.question_type}")
        if r.resolved_at is not None:
            lines.append(f"- Resolved at: {r.resolved_at.isoformat()}")
        if r.resolution is not None:
            lines.append(f"- Resolution: {r.resolution!r}")
        if r.my_forecast_at is not None:
            lines.append(f"- My last forecast at: {r.my_forecast_at.isoformat()}")
        lines.append(
            f"- My prediction: {_format_pred(r.my_prediction, question_type=r.question_type)}"
        )
        lines.append(
            f"- Community at my forecast: {_format_pred(r.community_at_my_forecast, question_type=r.question_type)}"
        )
        lines.append(
            f"- Community latest: {_format_pred(r.community_latest, question_type=r.question_type)}"
        )
        if r.score is not None:
            lines.append(f"- Score: {r.score:.4f}")
        if r.notes:
            lines.append(f"- Notes: {r.notes}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


async def run_retrospective(
    *,
    client: MetaculusClient,
    tournament_id: str,
    out_path: Path | None = None,
) -> str:
    timeout_seconds = _env_int(
        "BOT_RETROSPECTIVE_HTTP_TIMEOUT_SECONDS", _DEFAULT_HTTP_TIMEOUT_SECONDS
    )
    api2_max_retries = _env_int(
        "BOT_RETROSPECTIVE_API2_MAX_RETRIES", _DEFAULT_API2_MAX_RETRIES
    )
    api2_min_delay = _env_float(
        "BOT_RETROSPECTIVE_API2_MIN_DELAY_SECONDS", _DEFAULT_API2_MIN_DELAY_SECONDS
    )
    api2_backoff_base = _env_float(
        "BOT_RETROSPECTIVE_API2_BACKOFF_BASE_SECONDS",
        _DEFAULT_API2_BACKOFF_BASE_SECONDS,
    )
    max_questions = _env_int("BOT_RETROSPECTIVE_MAX_QUESTIONS", _DEFAULT_MAX_QUESTIONS)
    if max_questions <= 0:
        max_questions = _DEFAULT_MAX_QUESTIONS

    resolved = await _get_resolved_questions_i_forecasted(
        client=client, tournament_id=tournament_id, max_questions=max_questions
    )

    session = requests.Session()
    try:
        api2_cache: dict[int, dict | None] = {}
        rows: list[RetrospectiveRow] = []
        for question in resolved:
            if not bool(getattr(question, "already_forecasted", False)):
                continue
            question_id = getattr(question, "id_of_question", None)
            post_id = getattr(question, "id_of_post", None)
            if not isinstance(question_id, int) or not isinstance(post_id, int):
                continue

            if post_id in api2_cache:
                api2_post = api2_cache[post_id]
                if api2_post is None:
                    continue
            else:
                try:
                    api2_post = _fetch_api2_post(
                        session=session,
                        post_id=post_id,
                        timeout_seconds=timeout_seconds,
                        max_retries=api2_max_retries,
                        min_delay_seconds=api2_min_delay,
                        backoff_base_seconds=api2_backoff_base,
                    )
                except Exception:
                    api2_cache[post_id] = None
                    logger.info(
                        "Failed to fetch api2 payload for post_id=%s",
                        post_id,
                        exc_info=True,
                    )
                    continue
                api2_cache[post_id] = api2_post

            question_json = _select_api2_question_json(
                api2_post=api2_post, question_id=question_id
            )
            if not isinstance(question_json, dict):
                continue

            qtype = _question_type_from_json(question_json)
            title = _safe_title(question)
            url = _safe_url(question)

            my_time = _extract_last_forecast_time(question_json)
            my_pred = _extract_account_prediction_from_question_json(
                question_json=question_json, question_type=qtype
            )
            cp_latest = _extract_cp_latest(
                question_json=question_json, question_type=qtype
            )
            cp_at_my_time = (
                _extract_cp_at_time(
                    question_json=question_json, question_type=qtype, when=my_time
                )
                if my_time is not None
                else None
            )

            resolved_at = _parse_iso8601_utc(
                question_json.get("resolution_set_time")
                or question_json.get("actual_resolve_time")
                or question_json.get("actual_close_time")
            )
            resolution = question_json.get("resolution")

            score: float | None = None
            notes = ""
            if qtype == "binary":
                outcome = _resolution_to_binary_outcome(resolution)
                score = _score_binary(prediction=my_pred, outcome=outcome)
                if outcome is None:
                    notes = "binary_missing_outcome"
                elif score is None:
                    notes = "binary_missing_prediction"
                else:
                    # Heuristic notes to help triage misses.
                    if isinstance(cp_at_my_time, (int, float)) and isinstance(
                        my_pred, (int, float)
                    ):
                        abs_delta = abs(float(cp_at_my_time) - float(my_pred))
                        if abs_delta >= 0.20:
                            notes = f"contrarian_vs_cp(abs_delta={abs_delta:.2f})"
            elif qtype in {"numeric", "date", "discrete"}:
                # Partial scoring: report whether the resolution landed inside our 80% interval.
                x = _parse_float_maybe(resolution)
                if x is not None and isinstance(my_pred, dict):
                    percentile_map = _get_numeric_percentile_map(my_pred)
                    if percentile_map:
                        nearest_p10 = min(
                            percentile_map.keys(), key=lambda p: abs(p - 0.1)
                        )
                        nearest_p90 = min(
                            percentile_map.keys(), key=lambda p: abs(p - 0.9)
                        )
                        p10 = percentile_map.get(nearest_p10)
                        p90 = percentile_map.get(nearest_p90)
                    else:
                        p10 = p90 = None
                    if p10 is not None and p90 is not None:
                        inside = float(p10) <= float(x) <= float(p90)
                        notes = "inside_p10_p90" if inside else "outside_p10_p90"
            elif qtype == "multiple_choice":
                # Partial scoring: if we can match the resolved option to a probability in our map.
                if isinstance(resolution, str) and isinstance(my_pred, dict):
                    prob = my_pred.get(resolution)
                    if isinstance(prob, (int, float)):
                        notes = f"p(resolution)={float(prob):.3f}"

            rows.append(
                RetrospectiveRow(
                    question_id=question_id,
                    post_id=post_id,
                    url=url,
                    title=title,
                    question_type=qtype,
                    resolved_at=resolved_at,
                    resolution=resolution,
                    my_forecast_at=my_time,
                    my_prediction=my_pred,
                    community_at_my_forecast=cp_at_my_time,
                    community_latest=cp_latest,
                    score=score,
                    notes=notes,
                )
            )
    finally:
        session.close()

    markdown = _render_markdown(tournament_id=tournament_id, rows=rows)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(markdown, encoding="utf-8")
    return markdown
