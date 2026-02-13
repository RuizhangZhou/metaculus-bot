from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from forecasting_tools import MetaculusClient, MetaculusQuestion
from forecasting_tools.helpers.metaculus_client import ApiFilter

from metaculus_comment_fetcher import (
    extract_comment_text,
    fetch_my_comments,
    group_comments_by_post_id,
    select_comment_for_forecast_start_time,
)
from retrospective_mode import (
    _DEFAULT_HTTP_TIMEOUT_SECONDS,
    _extract_last_forecast_time,
    _format_pred,
    _parse_iso8601_utc,
    _question_type_from_json,
    _resolution_to_binary_outcome,
    _score_binary,
    _select_api2_question_json,
    _fetch_api2_post,
    _env_int,
    _env_float,
)
from tournament_update import _extract_cp_at_time, _extract_cp_latest

logger = logging.getLogger(__name__)


_DEFAULT_DAYS_LOOKBACK = 7
_DEFAULT_MAX_QUESTIONS_PER_TOURNAMENT = 100


@dataclass(frozen=True)
class WeeklyRetroRow:
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
    comment_text: str


def _key(post_id: int, question_id: int) -> str:
    return f"{post_id}:{question_id}"


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


def _load_processed_state(state_path: Path) -> dict[str, str]:
    try:
        if not state_path.exists():
            return {}
        data = json.loads(state_path.read_text(encoding="utf-8"))
        processed = data.get("processed")
        if isinstance(processed, dict):
            return {str(k): str(v) for k, v in processed.items() if isinstance(v, str)}
    except Exception:
        logger.info("Failed to load weekly retrospective state", exc_info=True)
    return {}


def _save_processed_state(state_path: Path, processed: dict[str, str]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "processed": dict(sorted(processed.items())),
    }
    state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _render_markdown(
    *, title: str, rows: list[WeeklyRetroRow], window_start: datetime, window_end: datetime
) -> str:
    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(
        f"Window: {window_start.date().isoformat()} to {window_end.date().isoformat()} (UTC)"
    )
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")

    if not rows:
        lines.append("No newly-resolved questions found in this window.")
        lines.append("")
        return "\n".join(lines)

    binary = [r for r in rows if r.question_type == "binary" and r.score is not None]
    if binary:
        avg_brier = sum(float(r.score) for r in binary) / len(binary)
        lines.append(f"- Binary: {len(binary)} scored, avg Brier={avg_brier:.4f}")
    lines.append(f"- Total items: {len(rows)}")
    lines.append("")

    lines.append("## Items")
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
        if r.comment_text:
            max_chars = _env_int("BOT_WEEKLY_RETRO_COMMENT_MAX_CHARS", 4000)
            text = r.comment_text.strip()
            if max_chars > 0 and len(text) > max_chars:
                text = text[: max_chars - 20].rstrip() + "\n\n[TRUNCATED]"
            lines.append("")
            lines.append("#### Bot explanation (from Metaculus comment)")
            lines.append("")
            lines.append("```markdown")
            lines.append(text)
            lines.append("```")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


async def run_weekly_retrospective(
    *,
    client: MetaculusClient,
    tournaments: list[str],
    out_path: Path,
    state_path: Path,
) -> str:
    days = _env_int("BOT_WEEKLY_RETRO_DAYS_LOOKBACK", _DEFAULT_DAYS_LOOKBACK)
    if days <= 0:
        days = _DEFAULT_DAYS_LOOKBACK
    window_end = datetime.now(timezone.utc)
    window_start = window_end - timedelta(days=days)

    max_per_tournament = _env_int(
        "BOT_WEEKLY_RETRO_MAX_QUESTIONS_PER_TOURNAMENT",
        _DEFAULT_MAX_QUESTIONS_PER_TOURNAMENT,
    )
    if max_per_tournament <= 0:
        max_per_tournament = _DEFAULT_MAX_QUESTIONS_PER_TOURNAMENT

    processed = _load_processed_state(state_path)
    updated_processed = dict(processed)

    token = os.getenv("METACULUS_TOKEN", "").strip()
    author_id = client.get_current_user_id()
    if not author_id:
        raise RuntimeError("Failed to determine Metaculus user id (author_id).")
    forecaster_id = int(author_id)

    timeout_seconds = _env_int(
        "BOT_RETROSPECTIVE_HTTP_TIMEOUT_SECONDS", _DEFAULT_HTTP_TIMEOUT_SECONDS
    )
    api2_max_retries = _env_int("BOT_RETROSPECTIVE_API2_MAX_RETRIES", 4)
    api2_min_delay = _env_float("BOT_RETROSPECTIVE_API2_MIN_DELAY_SECONDS", 0.2)
    api2_backoff_base = _env_float("BOT_RETROSPECTIVE_API2_BACKOFF_BASE_SECONDS", 1.0)

    api2_cache: dict[int, dict | None] = {}
    # Fetch all bot-authored comments once, then filter by post_id locally.
    # This avoids per-post API calls and works even if the endpoint ignores
    # `on_post` filters.
    all_comments: list[dict] = []
    max_comment_items = _env_int("BOT_WEEKLY_RETRO_MAX_TOTAL_COMMENTS", 500)
    include_private = _env_bool("BOT_RETRO_COMMENT_INCLUDE_PRIVATE", True)
    include_public = _env_bool("BOT_RETRO_COMMENT_INCLUDE_PUBLIC", True)
    try:
        if include_private:
            all_comments.extend(
                fetch_my_comments(
                    token=token,
                    author_id=int(author_id),
                    timeout_seconds=timeout_seconds,
                    max_items=max_comment_items,
                    include_private=True,
                )
            )
        if include_public:
            all_comments.extend(
                fetch_my_comments(
                    token=token,
                    author_id=int(author_id),
                    timeout_seconds=timeout_seconds,
                    max_items=max_comment_items,
                    include_private=False,
                )
            )
    except Exception:
        logger.info(
            "Failed to fetch bot comments; continuing without them", exc_info=True
        )

    # Deduplicate by comment id.
    unique_comments: dict[int, dict] = {}
    for item in all_comments:
        if not isinstance(item, dict):
            continue
        cid = item.get("id")
        if isinstance(cid, int):
            unique_comments[cid] = item
    comments_by_post_id = group_comments_by_post_id(list(unique_comments.values()))

    session = requests.Session()
    try:
        rows: list[WeeklyRetroRow] = []

        for tournament_id in tournaments:
            api_filter = ApiFilter(
                allowed_tournaments=[tournament_id],
                allowed_statuses=["resolved"],
                group_question_mode="unpack_subquestions",
                order_by="-actual_resolve_time",
                # Avoid repeated /api/users/me calls inside forecasting-tools by
                # passing the `forecaster_id` directly.
                other_url_parameters={"forecaster_id": forecaster_id},
            )

            # Paginate by default to avoid missing items when >1 page resolved in the
            # window. You can set BOT_WEEKLY_RETRO_FORCE_PAGINATION=0 to only fetch
            # the first page (rate-limit friendly, but may miss some resolved items).
            paginate = True
            if os.getenv("BOT_WEEKLY_RETRO_FORCE_PAGINATION") is not None:
                paginate = _env_int("BOT_WEEKLY_RETRO_FORCE_PAGINATION", 1) > 0
            num_questions = max_per_tournament if paginate else None

            questions = await client.get_questions_matching_filter(
                api_filter,
                num_questions=num_questions,
                randomly_sample=False,
                error_if_question_target_missed=False,
            )
            if max_per_tournament and len(questions) > max_per_tournament:
                questions = questions[:max_per_tournament]

            for question in questions:
                if not bool(getattr(question, "already_forecasted", False)):
                    continue

                # Fast prefilter to avoid api2 calls for older items.
                resolved_guess = getattr(question, "actual_resolution_time", None)
                if isinstance(resolved_guess, datetime):
                    resolved_guess_utc = resolved_guess.astimezone(timezone.utc)
                    if resolved_guess_utc < window_start:
                        break
                    if resolved_guess_utc > window_end:
                        continue

                question_id = getattr(question, "id_of_question", None)
                post_id = getattr(question, "id_of_post", None)
                if not isinstance(question_id, int) or not isinstance(post_id, int):
                    continue

                item_key = _key(post_id, question_id)

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

                resolved_at = _parse_iso8601_utc(
                    question_json.get("resolution_set_time")
                    or question_json.get("actual_resolve_time")
                    or question_json.get("actual_close_time")
                )
                if resolved_at is None:
                    continue
                if not (window_start <= resolved_at <= window_end):
                    continue

                prev = processed.get(item_key)
                if prev:
                    prev_dt = _parse_iso8601_utc(prev)
                    if prev_dt is not None and resolved_at <= prev_dt:
                        continue

                qtype = _question_type_from_json(question_json)
                title = _safe_title(question)
                url = _safe_url(question)

                my_time = _extract_last_forecast_time(question_json)
                from digest_mode import _extract_account_prediction_from_question_json

                my_prediction = _extract_account_prediction_from_question_json(
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

                resolution = question_json.get("resolution")
                score: float | None = None
                if qtype == "binary":
                    outcome = _resolution_to_binary_outcome(resolution)
                    score = _score_binary(prediction=my_prediction, outcome=outcome)

                comment = select_comment_for_forecast_start_time(
                    comments=comments_by_post_id.get(post_id, []),
                    forecast_start_time=my_time,
                )
                comment_text = extract_comment_text(comment)

                rows.append(
                    WeeklyRetroRow(
                        question_id=question_id,
                        post_id=post_id,
                        url=url,
                        title=title,
                        question_type=qtype,
                        resolved_at=resolved_at,
                        resolution=resolution,
                        my_forecast_at=my_time,
                        my_prediction=my_prediction,
                        community_at_my_forecast=cp_at_my_time,
                        community_latest=cp_latest,
                        score=score,
                        comment_text=comment_text,
                    )
                )
                updated_processed[item_key] = resolved_at.isoformat()

        rows.sort(
            key=lambda r: r.resolved_at or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )

        markdown = _render_markdown(
            title="Weekly retrospective (resolved questions)",
            rows=rows,
            window_start=window_start,
            window_end=window_end,
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(markdown, encoding="utf-8")
        _save_processed_state(state_path, updated_processed)
        return markdown
    finally:
        session.close()
