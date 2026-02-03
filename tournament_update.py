from __future__ import annotations

import logging
from datetime import datetime, timezone

from forecasting_tools import MetaculusClient, MetaculusQuestion

from digest_mode import _get_close_time_iso, _is_significant_change

logger = logging.getLogger(__name__)


def _ensure_timezone(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _question_type_string(question: MetaculusQuestion) -> str:
    question_type = getattr(question, "question_type", None)
    if isinstance(question_type, str):
        return question_type
    return "unknown"


def _select_aggregation_block(question_json: dict) -> dict | None:
    aggregations = question_json.get("aggregations")
    if not isinstance(aggregations, dict):
        return None
    for key in ("recency_weighted", "unweighted"):
        block = aggregations.get(key)
        if isinstance(block, dict):
            return block
    return None


def _extract_options(question_json: dict) -> list[str]:
    options_raw = question_json.get("options")
    options: list[str] = []
    if isinstance(options_raw, list):
        for opt in options_raw:
            if isinstance(opt, str):
                options.append(opt)
            elif isinstance(opt, dict):
                name = opt.get("name") or opt.get("label") or opt.get("title")
                if isinstance(name, str):
                    options.append(name)
    return options


def _continuous_range_from_scaling(
    *, question_json: dict, num_points: int
) -> list[float] | None:
    scaling = question_json.get("scaling")
    if not isinstance(scaling, dict):
        return None

    continuous_range_raw = scaling.get("continuous_range")
    if isinstance(continuous_range_raw, list):
        continuous_range = [
            float(x)
            for x in continuous_range_raw
            if isinstance(x, (int, float))
        ]
        if len(continuous_range) == num_points:
            return continuous_range

    range_min = scaling.get("range_min")
    range_max = scaling.get("range_max")
    if (
        isinstance(range_min, (int, float))
        and isinstance(range_max, (int, float))
        and num_points > 1
    ):
        step = (float(range_max) - float(range_min)) / (num_points - 1)
        return [float(range_min) + step * i for i in range(num_points)]

    return None


def _extract_float_list(value: object) -> list[float] | None:
    if not isinstance(value, list):
        return None
    out: list[float] = []
    for item in value:
        if not isinstance(item, (int, float)):
            return None
        out.append(float(item))
    return out


def _extract_cp_from_aggregation_item(
    *,
    aggregation_item: dict,
    question_json: dict,
    question_type: str,
) -> object | None:
    if question_type == "binary":
        centers = _extract_float_list(aggregation_item.get("centers"))
        if centers and len(centers) == 1:
            return float(centers[0])
        means = _extract_float_list(aggregation_item.get("means"))
        if means and len(means) == 1:
            return float(means[0])
        forecast_values = _extract_float_list(aggregation_item.get("forecast_values"))
        if forecast_values and len(forecast_values) >= 2:
            return float(forecast_values[1])
        return None

    if question_type == "multiple_choice":
        per_category = aggregation_item.get("probability_yes_per_category")
        if isinstance(per_category, dict):
            mapped = {
                str(k): float(v)
                for k, v in per_category.items()
                if isinstance(k, str) and isinstance(v, (int, float))
            }
            return mapped or None

        forecast_values = _extract_float_list(aggregation_item.get("forecast_values"))
        if forecast_values is None:
            forecast_values = _extract_float_list(aggregation_item.get("centers"))

        if not forecast_values:
            return None

        options = _extract_options(question_json)
        if options and len(options) == len(forecast_values):
            mapped = {
                option: float(prob)
                for option, prob in zip(options, forecast_values)
                if isinstance(option, str)
            }
            return mapped or None
        return None

    if question_type in {"numeric", "date", "discrete"}:
        centers = _extract_float_list(aggregation_item.get("centers"))
        if centers and len(centers) == 1:
            center = float(centers[0])
            lower_bounds = _extract_float_list(
                aggregation_item.get("interval_lower_bounds")
            )
            upper_bounds = _extract_float_list(
                aggregation_item.get("interval_upper_bounds")
            )
            declared: list[dict[str, float]] = [{"percentile": 0.5, "value": center}]
            if (
                lower_bounds
                and upper_bounds
                and len(lower_bounds) == 1
                and len(upper_bounds) == 1
            ):
                declared.insert(0, {"percentile": 0.1, "value": float(lower_bounds[0])})
                declared.append({"percentile": 0.9, "value": float(upper_bounds[0])})
            return {"declared_percentiles": declared}

        forecast_values = _extract_float_list(aggregation_item.get("forecast_values"))
        if forecast_values is None:
            forecast_values = _extract_float_list(aggregation_item.get("continuous_cdf"))
        if not forecast_values:
            return None

        continuous_range = _continuous_range_from_scaling(
            question_json=question_json, num_points=len(forecast_values)
        )
        if not continuous_range or len(continuous_range) != len(forecast_values):
            return None

        declared_percentiles: list[dict[str, float]] = []
        for x, cdf in zip(continuous_range, forecast_values):
            if not (0.0 <= float(cdf) <= 1.0):
                continue
            declared_percentiles.append(
                {"percentile": float(cdf), "value": float(x)}
            )
        return {"declared_percentiles": declared_percentiles} if declared_percentiles else None

    return None


def _extract_cp_latest(
    *, question_json: dict, question_type: str
) -> object | None:
    aggregation_block = _select_aggregation_block(question_json)
    if not aggregation_block:
        return None
    latest = aggregation_block.get("latest")
    if not isinstance(latest, dict):
        return None
    return _extract_cp_from_aggregation_item(
        aggregation_item=latest,
        question_json=question_json,
        question_type=question_type,
    )


def _extract_cp_at_time(
    *, question_json: dict, question_type: str, when: datetime
) -> object | None:
    aggregation_block = _select_aggregation_block(question_json)
    if not aggregation_block:
        return None
    history = aggregation_block.get("history")
    if not isinstance(history, list) or not history:
        return None

    ts = _ensure_timezone(when).timestamp()
    best_before: dict | None = None
    best_before_start: float | None = None
    for item in history:
        if not isinstance(item, dict):
            continue
        start_time = item.get("start_time")
        end_time = item.get("end_time")
        if not isinstance(start_time, (int, float)):
            continue
        if start_time <= ts and (
            not end_time or (isinstance(end_time, (int, float)) and ts <= end_time)
        ):
            return _extract_cp_from_aggregation_item(
                aggregation_item=item,
                question_json=question_json,
                question_type=question_type,
            )
        if start_time <= ts and (
            best_before_start is None or float(start_time) > best_before_start
        ):
            best_before = item
            best_before_start = float(start_time)
    if best_before is not None:
        return _extract_cp_from_aggregation_item(
            aggregation_item=best_before,
            question_json=question_json,
            question_type=question_type,
        )
    for item in history:
        if isinstance(item, dict) and isinstance(item.get("start_time"), (int, float)):
            return _extract_cp_from_aggregation_item(
                aggregation_item=item,
                question_json=question_json,
                question_type=question_type,
            )
    return None


def _question_json_from_question(question: MetaculusQuestion) -> dict | None:
    api_json = getattr(question, "api_json", None)
    if not isinstance(api_json, dict):
        return None
    question_json = api_json.get("question")
    return question_json if isinstance(question_json, dict) else None


def _get_full_question_json(
    *,
    client: MetaculusClient,
    post_id: int,
    question_id: int,
    cache: dict[int, dict[int, dict]],
) -> dict | None:
    by_question_id = cache.get(post_id)
    if isinstance(by_question_id, dict) and question_id in by_question_id:
        return by_question_id[question_id]

    fetched = client.get_question_by_post_id(
        post_id, group_question_mode="unpack_subquestions"
    )
    fetched_questions = fetched if isinstance(fetched, list) else [fetched]
    mapping: dict[int, dict] = {}
    for q in fetched_questions:
        qid = getattr(q, "id_of_question", None)
        if not isinstance(qid, int):
            continue
        qjson = _question_json_from_question(q)
        if isinstance(qjson, dict):
            mapping[qid] = qjson
    cache[post_id] = mapping
    return mapping.get(question_id)


def select_questions_for_tournament_update(
    *,
    client: MetaculusClient,
    tournament_id: str,
    max_questions: int | None = None,
) -> tuple[list[MetaculusQuestion], dict[str, int]]:
    """
    Returns questions that should be re-forecasted:
    - Any open question the bot has never forecasted
    - Any open question where community prediction has shifted significantly since the bot's last forecast

    This relies on Metaculus aggregation history and the bot's "my_forecasts.latest.start_time".
    No local state is required (works well on ephemeral GitHub Actions runners).
    """

    all_questions = client.get_all_open_questions_from_tournament(tournament_id)
    questions = (
        all_questions[:max_questions] if max_questions is not None else all_questions
    )
    full_question_cache: dict[int, dict[int, dict]] = {}

    counts: dict[str, int] = {
        "total_open": len(questions),
        "queued_unforecasted": 0,
        "queued_cp_changed": 0,
        "skipped_no_last_forecast_time": 0,
        "skipped_missing_cp": 0,
        "skipped_other": 0,
    }
    selected: list[MetaculusQuestion] = []

    for question in questions:
        already_forecasted = bool(getattr(question, "already_forecasted", False))
        if not already_forecasted:
            selected.append(question)
            counts["queued_unforecasted"] += 1
            continue

        last_forecast_time = getattr(question, "timestamp_of_my_last_forecast", None)
        if not isinstance(last_forecast_time, datetime):
            counts["skipped_no_last_forecast_time"] += 1
            continue

        question_id = getattr(question, "id_of_question", None)
        post_id = getattr(question, "id_of_post", None)
        if not isinstance(question_id, int) or not isinstance(post_id, int):
            counts["skipped_other"] += 1
            continue

        qtype = _question_type_string(question)
        close_time_iso = _get_close_time_iso(question)

        question_json = _question_json_from_question(question)
        if not isinstance(question_json, dict):
            counts["skipped_other"] += 1
            continue

        old_cp = _extract_cp_at_time(
            question_json=question_json,
            question_type=qtype,
            when=last_forecast_time,
        )
        new_cp = _extract_cp_latest(question_json=question_json, question_type=qtype)

        if old_cp is None:
            full_question_json = _get_full_question_json(
                client=client,
                post_id=post_id,
                question_id=question_id,
                cache=full_question_cache,
            )
            if isinstance(full_question_json, dict):
                question_json = full_question_json
                old_cp = _extract_cp_at_time(
                    question_json=question_json,
                    question_type=qtype,
                    when=last_forecast_time,
                )
                new_cp = _extract_cp_latest(
                    question_json=question_json, question_type=qtype
                )
        if old_cp is None or new_cp is None:
            counts["skipped_missing_cp"] += 1
            continue

        significant, reason = _is_significant_change(
            old_pred=old_cp,
            new_pred=new_cp,
            question_type=qtype,
            close_time_iso=close_time_iso,
            old_close_time_iso=close_time_iso,
        )
        if significant:
            selected.append(question)
            counts["queued_cp_changed"] += 1
            url = getattr(question, "page_url", None) or f"post:{getattr(question, 'id_of_post', '')}"
            logger.info(f"Queueing update ({reason}) for {url}")

    return selected, counts
