import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import requests

from forecasting_tools import (
    BinaryQuestion,
    ConditionalQuestion,
    DateQuestion,
    ForecastBot,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)

logger = logging.getLogger(__name__)


def _normalize_metaculus_token(token: str | None) -> str:
    if token is None:
        return ""
    stripped = token.strip()
    lowered = stripped.lower()
    if lowered.startswith("token "):
        stripped = stripped.split(None, 1)[1].strip()
    elif lowered.startswith("bearer "):
        stripped = stripped.split(None, 1)[1].strip()
    return stripped


def extract_tournament_identifier(value: str) -> str | None:
    raw = value.strip()
    if not raw or raw.startswith("#"):
        return None

    if raw.startswith("http://") or raw.startswith("https://"):
        tournament_match = re.search(r"/tournament/([^/]+)/?", raw)
        if tournament_match:
            slug = tournament_match.group(1).strip("/")
            return slug if slug.isdigit() else slug.lower()
        index_match = re.search(r"/index/([^/]+)/?", raw)
        if index_match:
            return f"index:{index_match.group(1)}"
        return None

    slug = raw.strip("/")
    return slug if slug.isdigit() else slug.lower()


def load_tournament_identifiers(
    tournaments_file: str | None, extra_identifiers: list[str] | None
) -> tuple[list[str], list[str]]:
    identifiers: list[str] = []
    unsupported: list[str] = []

    if tournaments_file:
        path = Path(tournaments_file)
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                identifier = extract_tournament_identifier(line)
                if not identifier:
                    continue
                if identifier.startswith("index:"):
                    unsupported.append(identifier)
                else:
                    identifiers.append(identifier)
        else:
            logger.warning(f"Tournaments file not found: {tournaments_file}")

    for raw in extra_identifiers or []:
        identifier = extract_tournament_identifier(raw)
        if not identifier:
            continue
        if identifier.startswith("index:"):
            unsupported.append(identifier)
        else:
            identifiers.append(identifier)

    seen: set[str] = set()
    unique_identifiers: list[str] = []
    for identifier in identifiers:
        if identifier in seen:
            continue
        seen.add(identifier)
        unique_identifiers.append(identifier)

    return unique_identifiers, unsupported


def _prediction_to_compact_jsonable(prediction: object) -> object:
    if isinstance(prediction, (float, int, str, bool)) or prediction is None:
        return prediction
    model_dump = getattr(prediction, "model_dump", None)
    if callable(model_dump):
        return model_dump()
    to_dict = getattr(prediction, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    as_dict = getattr(prediction, "dict", None)
    if callable(as_dict):
        return as_dict()
    return str(prediction)


def _get_close_time_iso(question: MetaculusQuestion) -> str | None:
    close_time = getattr(question, "close_time", None)
    if close_time is None:
        return None
    if isinstance(close_time, datetime):
        return close_time.astimezone(timezone.utc).isoformat()
    return str(close_time)


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _days_until(close_time_iso: str | None) -> float | None:
    dt = _parse_iso_datetime(close_time_iso)
    if dt is None:
        return None
    return (dt - datetime.now(timezone.utc)).total_seconds() / 86400


def _threshold_factor_from_days_left(days_left: float | None) -> float:
    if days_left is None:
        return 1.0
    if days_left <= 2:
        return 0.5
    if days_left <= 7:
        return 0.7
    return 1.0


def _get_numeric_percentile_map(pred: dict) -> dict[float, float]:
    percentiles = pred.get("declared_percentiles")
    if not isinstance(percentiles, list):
        return {}
    out: dict[float, float] = {}
    for p in percentiles:
        if not isinstance(p, dict):
            continue
        perc = p.get("percentile")
        val = p.get("value")
        if isinstance(perc, (int, float)) and isinstance(val, (int, float)):
            out[float(perc)] = float(val)
    return out


def _approx_median_from_percentiles(pmap: dict[float, float]) -> float | None:
    if 0.5 in pmap:
        return pmap[0.5]
    p40 = pmap.get(0.4)
    p60 = pmap.get(0.6)
    if p40 is not None and p60 is not None:
        return 0.5 * (p40 + p60)
    if not pmap:
        return None
    nearest = min(pmap.keys(), key=lambda p: abs(p - 0.5))
    return pmap[nearest]


def _is_significant_change(
    *,
    old_pred: object | None,
    new_pred: object | None,
    question_type: str,
    close_time_iso: str | None,
    old_close_time_iso: str | None,
) -> tuple[bool, str]:
    days_left = _days_until(close_time_iso)
    factor = _threshold_factor_from_days_left(days_left)

    if old_pred is None and new_pred is not None:
        return True, "new_question"
    if old_pred is not None and new_pred is None:
        return False, "question_missing_now"

    if close_time_iso and old_close_time_iso and close_time_iso != old_close_time_iso:
        new_dt = _parse_iso_datetime(close_time_iso)
        old_dt = _parse_iso_datetime(old_close_time_iso)
        if new_dt and old_dt:
            delta_hours = abs((new_dt - old_dt).total_seconds()) / 3600
            if delta_hours >= 24:
                return True, f"close_time_changed_{delta_hours:.1f}h"

    if question_type == "binary":
        if not isinstance(old_pred, (int, float)) or not isinstance(
            new_pred, (int, float)
        ):
            return True, "binary_type_changed"
        old_p = float(old_pred)
        new_p = float(new_pred)
        base = 0.10
        threshold = base * factor
        abs_delta = abs(new_p - old_p)
        crossed = (old_p - 0.5) * (new_p - 0.5) < 0
        if abs_delta >= threshold:
            return True, f"abs_delta={abs_delta:.3f}"
        if crossed and abs_delta >= max(0.04, 0.5 * threshold):
            return True, f"crossed_50_abs_delta={abs_delta:.3f}"
        return False, f"abs_delta={abs_delta:.3f}"

    if question_type == "multiple_choice":
        if not isinstance(old_pred, dict) or not isinstance(new_pred, dict):
            return True, "mc_type_changed"

        def to_map(d: dict) -> dict[str, float]:
            if "predicted_options" in d and isinstance(d["predicted_options"], list):
                m: dict[str, float] = {}
                for item in d["predicted_options"]:
                    if not isinstance(item, dict):
                        continue
                    name = item.get("option_name")
                    prob = item.get("probability")
                    if isinstance(name, str) and isinstance(prob, (int, float)):
                        m[name] = float(prob)
                return m
            return {
                k: float(v)
                for k, v in d.items()
                if isinstance(k, str) and isinstance(v, (int, float))
            }

        old_map = to_map(old_pred)
        new_map = to_map(new_pred)
        keys = sorted(set(old_map) | set(new_map))
        tvd = 0.5 * sum(
            abs(new_map.get(k, 0.0) - old_map.get(k, 0.0)) for k in keys
        )
        base = 0.15
        threshold = base * factor
        if tvd >= threshold:
            return True, f"tvd={tvd:.3f}"
        return False, f"tvd={tvd:.3f}"

    if question_type in {"numeric", "date", "discrete"}:
        if not isinstance(old_pred, dict) or not isinstance(new_pred, dict):
            return True, "numeric_type_changed"
        old_map = _get_numeric_percentile_map(old_pred)
        new_map = _get_numeric_percentile_map(new_pred)
        old_med = _approx_median_from_percentiles(old_map)
        new_med = _approx_median_from_percentiles(new_map)
        if old_med is None or new_med is None:
            return True, "missing_median"
        old_p10 = old_map.get(0.1)
        old_p90 = old_map.get(0.9)
        new_p10 = new_map.get(0.1)
        new_p90 = new_map.get(0.9)
        old_width = (
            (old_p90 - old_p10)
            if (old_p10 is not None and old_p90 is not None)
            else None
        )
        new_width = (
            (new_p90 - new_p10)
            if (new_p10 is not None and new_p90 is not None)
            else None
        )
        width: float
        if old_width is not None and new_width is not None:
            width = max(1e-9, 0.5 * (old_width + new_width))
        elif old_width is not None:
            width = max(1e-9, old_width)
        elif new_width is not None:
            width = max(1e-9, new_width)
        else:
            width = max(1e-9, abs(old_med), abs(new_med))

        normalized_median_shift = abs(new_med - old_med) / width
        base = 0.35
        threshold = base * factor

        if question_type == "date":
            median_shift_days = abs(new_med - old_med) / 86400
            absolute_day_threshold = 30 * factor
            if median_shift_days >= absolute_day_threshold:
                return True, f"median_shift_days={median_shift_days:.1f}"

        if normalized_median_shift >= threshold:
            return True, f"norm_median_shift={normalized_median_shift:.3f}"
        return False, f"norm_median_shift={normalized_median_shift:.3f}"

    if question_type == "conditional":
        if not isinstance(old_pred, dict) or not isinstance(new_pred, dict):
            return True, "conditional_type_changed"
        old_child = old_pred.get("child")
        new_child = new_pred.get("child")
        child_type = "unknown"
        if isinstance(new_child, (int, float)):
            child_type = "binary"
        elif isinstance(new_child, dict) and "predicted_options" in new_child:
            child_type = "multiple_choice"
        elif isinstance(new_child, dict) and "declared_percentiles" in new_child:
            child_type = "numeric"
        return _is_significant_change(
            old_pred=old_child,
            new_pred=new_child,
            question_type=child_type,
            close_time_iso=close_time_iso,
            old_close_time_iso=old_close_time_iso,
        )

    return True, f"unknown_type_{question_type}"


def matrix_send_message(message: str) -> None:
    homeserver = os.getenv("MATRIX_HOMESERVER")
    access_token = os.getenv("MATRIX_ACCESS_TOKEN")
    room_id = os.getenv("MATRIX_ROOM_ID")
    if not homeserver or not access_token or not room_id:
        return

    txn_id = uuid4().hex
    room_id_escaped = requests.utils.quote(room_id, safe="")
    url = (
        f"{homeserver.rstrip('/')}/_matrix/client/v3/rooms/"
        f"{room_id_escaped}/send/m.room.message/{txn_id}"
    )
    headers = {"Authorization": f"Bearer {access_token}"}
    payload = {"msgtype": "m.text", "body": message}
    response = requests.put(url, headers=headers, json=payload, timeout=30)
    if not response.ok:
        logger.warning(f"Matrix send failed: {response.status_code} {response.text}")


def _metaculus_get_post_json(*, post_id: int, token: str, timeout: int = 30) -> dict:
    url = f"https://www.metaculus.com/api/posts/{post_id}/"
    response = requests.get(
        url,
        headers={"Authorization": f"Token {token}"},
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected post JSON type for {post_id}: {type(data)}")
    return data


def _select_question_json_from_post_json(
    post_json: dict, question_id: int | None
) -> dict | None:
    question = post_json.get("question")
    if not isinstance(question, dict):
        return None

    if question_id is None or question.get("id") == question_id:
        return question

    group = post_json.get("group_of_questions")
    if isinstance(group, dict):
        group_questions = group.get("questions")
        if isinstance(group_questions, list):
            for q in group_questions:
                if isinstance(q, dict) and q.get("id") == question_id:
                    return q

    for key in ("question_yes", "question_no", "condition", "condition_child"):
        nested = question.get(key)
        if isinstance(nested, dict) and nested.get("id") == question_id:
            return nested

    return question


def _latest_forecast_entry(my_forecasts: object) -> dict | None:
    if not isinstance(my_forecasts, dict):
        return None

    latest = my_forecasts.get("latest")
    if isinstance(latest, dict):
        return latest

    history = my_forecasts.get("history")
    if isinstance(history, list) and history:
        last = history[-1]
        if isinstance(last, dict):
            return last

    return None


def _extract_account_prediction_from_question_json(
    *, question_json: dict, question_type: str
) -> object | None:
    latest = _latest_forecast_entry(question_json.get("my_forecasts"))
    if not isinstance(latest, dict):
        return None

    forecast_values = latest.get("forecast_values")

    if question_type == "binary":
        probability_yes = latest.get("probability_yes")
        if isinstance(probability_yes, (int, float)):
            return float(probability_yes)
        if (
            isinstance(forecast_values, list)
            and len(forecast_values) >= 2
            and isinstance(forecast_values[1], (int, float))
        ):
            return float(forecast_values[1])
        return None

    if question_type == "multiple_choice":
        per_category = latest.get("probability_yes_per_category")
        if isinstance(per_category, dict):
            mapped = {
                str(k): float(v)
                for k, v in per_category.items()
                if isinstance(k, str) and isinstance(v, (int, float))
            }
            return mapped or None

        if not isinstance(forecast_values, list):
            return None

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

        if options and len(options) == len(forecast_values):
            mapped = {
                option: float(prob)
                for option, prob in zip(options, forecast_values)
                if isinstance(option, str) and isinstance(prob, (int, float))
            }
            return mapped or None
        return None

    if question_type in {"numeric", "date", "discrete"}:
        if not isinstance(forecast_values, list) or not forecast_values:
            return None

        scaling = question_json.get("scaling")
        continuous_range: list[float] | None = None
        if isinstance(scaling, dict) and isinstance(scaling.get("continuous_range"), list):
            continuous_range = [
                float(x) for x in scaling["continuous_range"] if isinstance(x, (int, float))
            ]
        elif isinstance(scaling, dict):
            range_min = scaling.get("range_min")
            range_max = scaling.get("range_max")
            if (
                isinstance(range_min, (int, float))
                and isinstance(range_max, (int, float))
                and len(forecast_values) > 1
            ):
                step = (float(range_max) - float(range_min)) / (len(forecast_values) - 1)
                continuous_range = [float(range_min) + step * i for i in range(len(forecast_values))]

        if (
            not isinstance(continuous_range, list)
            or len(continuous_range) != len(forecast_values)
        ):
            return None

        declared_percentiles: list[dict[str, float]] = []
        for x, cdf in zip(continuous_range, forecast_values):
            if not isinstance(cdf, (int, float)):
                continue
            declared_percentiles.append({"percentile": float(cdf), "value": float(x)})
        return {"declared_percentiles": declared_percentiles} if declared_percentiles else None

    return None


def _extract_account_prediction_from_post_json(
    *, post_json: dict, question_id: int | None, question_type: str
) -> object | None:
    question_json = _select_question_json_from_post_json(post_json, question_id)
    if not isinstance(question_json, dict):
        return None
    return _extract_account_prediction_from_question_json(
        question_json=question_json, question_type=question_type
    )


def _approx_value_at_percentile(pmap: dict[float, float], target: float) -> float | None:
    if not pmap:
        return None
    nearest = min(pmap.keys(), key=lambda p: abs(p - target))
    return pmap[nearest]


def _format_prediction_for_markdown(*, pred: object | None, question_type: str) -> str:
    if pred is None:
        return "(none)"

    if question_type == "binary" and isinstance(pred, (int, float)):
        return f"{float(pred) * 100:.1f}%"

    if question_type == "multiple_choice" and isinstance(pred, dict):
        m: dict[str, float] = {}
        if "predicted_options" in pred and isinstance(pred["predicted_options"], list):
            for item in pred["predicted_options"]:
                if not isinstance(item, dict):
                    continue
                name = item.get("option_name")
                prob = item.get("probability")
                if isinstance(name, str) and isinstance(prob, (int, float)):
                    m[name] = float(prob)
        else:
            m = {
                k: float(v)
                for k, v in pred.items()
                if isinstance(k, str) and isinstance(v, (int, float))
            }
        if m:
            top = sorted(m.items(), key=lambda kv: kv[1], reverse=True)[:5]
            return ", ".join([f"{k}={v*100:.1f}%" for k, v in top])
        return "(unparseable multiple-choice prediction)"

    if question_type in {"numeric", "date", "discrete"} and isinstance(pred, dict):
        pmap = _get_numeric_percentile_map(pred)
        p10 = _approx_value_at_percentile(pmap, 0.1)
        p50 = _approx_value_at_percentile(pmap, 0.5)
        p90 = _approx_value_at_percentile(pmap, 0.9)
        if p50 is None:
            return "(missing numeric median)"
        if question_type == "date":
            def fmt_ts(ts: float | None) -> str:
                if ts is None:
                    return "?"
                try:
                    return datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat()
                except Exception:
                    return str(ts)

            return f"p50≈{fmt_ts(p50)} (p10≈{fmt_ts(p10)}, p90≈{fmt_ts(p90)})"

        def fmt_num(x: float | None) -> str:
            if x is None:
                return "?"
            return f"{x:.4g}"

        return f"p50≈{fmt_num(p50)} (p10≈{fmt_num(p10)}, p90≈{fmt_num(p90)})"

    try:
        text = json.dumps(pred, ensure_ascii=False)
    except Exception:
        text = str(pred)
    if len(text) > 500:
        return text[:500] + "…"
    return text


async def run_digest(
    *,
    template_bot: ForecastBot,
    tournaments: list[str],
    state_path: Path,
    out_dir: Path,
) -> None:
    from forecasting_tools.data_models.data_organizer import DataOrganizer

    compare_mode = os.getenv("DIGEST_COMPARE_MODE", "account").strip().lower()
    if compare_mode not in {"account", "state"}:
        logger.warning(f"Unknown DIGEST_COMPARE_MODE={compare_mode!r}; using 'account'")
        compare_mode = "account"

    max_questions_per_tournament: int | None = None
    max_questions_raw = os.getenv("DIGEST_MAX_QUESTIONS_PER_TOURNAMENT")
    if max_questions_raw is not None and max_questions_raw.strip():
        try:
            max_questions_per_tournament = int(max_questions_raw.strip())
        except ValueError:
            logger.warning(
                f"Ignoring invalid DIGEST_MAX_QUESTIONS_PER_TOURNAMENT={max_questions_raw!r}"
            )
            max_questions_per_tournament = None

    track_token = _normalize_metaculus_token(os.getenv("METACULUS_TRACK_TOKEN"))
    if track_token and any(ch.isspace() for ch in track_token):
        logger.warning("METACULUS_TRACK_TOKEN contains whitespace; ignoring it.")
        track_token = ""

    out_dir.mkdir(parents=True, exist_ok=True)
    state_path.parent.mkdir(parents=True, exist_ok=True)

    previous_state: dict = {}
    if state_path.exists():
        try:
            previous_state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Failed to read previous state: {e}")
            previous_state = {}

    previous_questions: dict = (
        previous_state.get("questions", {}) if isinstance(previous_state, dict) else {}
    )
    now_iso = datetime.now(timezone.utc).isoformat()

    new_state_questions: dict[str, dict] = {}
    significant_changes: list[dict] = []
    significant_changes_by_account: dict[str, list[dict]] = {"bot": [], "track": []}
    unforecasted_by_account: dict[str, list[dict]] = {"bot": [], "track": []}
    failures: list[dict] = []
    track_post_cache: dict[int, dict] = {}

    def safe_report_attr(report: object, attr: str) -> str | None:
        try:
            return getattr(report, attr)
        except Exception:
            return None

    for tournament_id in tournaments:
        try:
            if max_questions_per_tournament is None:
                reports_or_errors = await template_bot.forecast_on_tournament(
                    tournament_id, return_exceptions=True
                )
            else:
                from forecasting_tools import MetaculusClient

                client = MetaculusClient()
                questions = client.get_all_open_questions_from_tournament(tournament_id)
                if max_questions_per_tournament <= 0:
                    questions = []
                else:
                    questions = questions[:max_questions_per_tournament]
                reports_or_errors = await template_bot.forecast_questions(
                    questions, return_exceptions=True
                )
        except Exception as e:
            failures.append({"tournament": tournament_id, "error": str(e)})
            continue

        for item in reports_or_errors:
            if isinstance(item, BaseException):
                failures.append({"tournament": tournament_id, "error": repr(item)})
                continue

            report = item
            question = report.question
            key = "|".join(
                [
                    str(question.id_of_post or ""),
                    str(question.id_of_question or ""),
                    str(question.conditional_type or ""),
                    str(question.group_question_option or ""),
                ]
            )

            if isinstance(question, BinaryQuestion):
                qtype = "binary"
            elif isinstance(question, MultipleChoiceQuestion):
                qtype = "multiple_choice"
            elif isinstance(question, DateQuestion):
                qtype = "date"
            elif isinstance(question, NumericQuestion):
                qtype = "numeric"
            elif isinstance(question, ConditionalQuestion):
                qtype = "conditional"
            else:
                qtype = "unknown"

            close_time_iso = _get_close_time_iso(question)
            prediction_jsonable = _prediction_to_compact_jsonable(report.prediction)
            try:
                readable_prediction = DataOrganizer.get_readable_prediction(
                    report.prediction
                )
            except Exception as e:
                readable_prediction = f"(failed to format prediction: {e})"

            state_snapshot = {
                "generated_at": now_iso,
                "tournament": tournament_id,
                "question_text": question.question_text,
                "page_url": question.page_url,
                "id_of_post": question.id_of_post,
                "id_of_question": question.id_of_question,
                "close_time": close_time_iso,
                "question_type": qtype,
                "prediction": prediction_jsonable,
            }
            new_state_questions[key] = state_snapshot

            if compare_mode == "state":
                old_snapshot = previous_questions.get(key)
                old_pred = (
                    old_snapshot.get("prediction")
                    if isinstance(old_snapshot, dict)
                    else None
                )
                old_close_time = (
                    old_snapshot.get("close_time")
                    if isinstance(old_snapshot, dict)
                    else None
                )

                significant, reason = _is_significant_change(
                    old_pred=old_pred,
                    new_pred=prediction_jsonable,
                    question_type=qtype,
                    close_time_iso=close_time_iso,
                    old_close_time_iso=old_close_time,
                )
                if significant:
                    significant_changes.append(
                        {
                            "key": key,
                            "tournament": tournament_id,
                            "page_url": question.page_url,
                            "question_text": question.question_text,
                            "question_type": qtype,
                            "close_time": close_time_iso,
                            "reason": reason,
                            "old_prediction": old_pred,
                            "new_prediction": prediction_jsonable,
                            "readable_prediction": readable_prediction,
                            "summary": safe_report_attr(report, "summary"),
                            "research": safe_report_attr(report, "research"),
                            "forecast_rationales": safe_report_attr(
                                report, "forecast_rationales"
                            ),
                            "explanation": getattr(report, "explanation", None),
                        }
                    )
                continue

            bot_baseline = _extract_account_prediction_from_post_json(
                post_json=getattr(question, "api_json", {}) or {},
                question_id=question.id_of_question,
                question_type=qtype,
            )

            track_baseline: object | None = None
            if track_token and question.id_of_post:
                post_id = int(question.id_of_post)
                post_json_for_track = track_post_cache.get(post_id)
                if post_json_for_track is None:
                    try:
                        post_json_for_track = _metaculus_get_post_json(
                            post_id=post_id, token=track_token
                        )
                        track_post_cache[post_id] = post_json_for_track
                    except Exception as e:
                        failures.append(
                            {
                                "tournament": tournament_id,
                                "post_id": post_id,
                                "error": f"Failed to fetch post as track account: {e}",
                            }
                        )
                        post_json_for_track = None
                if isinstance(post_json_for_track, dict):
                    track_baseline = _extract_account_prediction_from_post_json(
                        post_json=post_json_for_track,
                        question_id=question.id_of_question,
                        question_type=qtype,
                    )

            comparisons = [
                ("bot", bot_baseline),
            ]
            if track_token:
                comparisons.append(("track", track_baseline))

            for account_name, baseline in comparisons:
                if baseline is None:
                    unforecasted_by_account[account_name].append(
                        {
                            "key": key,
                            "tournament": tournament_id,
                            "page_url": question.page_url,
                            "question_text": question.question_text,
                            "question_type": qtype,
                            "close_time": close_time_iso,
                        }
                    )
                    continue

                significant, reason = _is_significant_change(
                    old_pred=baseline,
                    new_pred=prediction_jsonable,
                    question_type=qtype,
                    close_time_iso=close_time_iso,
                    old_close_time_iso=None,
                )
                if not significant:
                    continue

                change = {
                    "key": key,
                    "account": account_name,
                    "tournament": tournament_id,
                    "page_url": question.page_url,
                    "question_text": question.question_text,
                    "question_type": qtype,
                    "close_time": close_time_iso,
                    "reason": reason,
                    "account_prediction": baseline,
                    "ai_prediction": prediction_jsonable,
                    "readable_prediction": readable_prediction,
                    "summary": safe_report_attr(report, "summary"),
                    "research": safe_report_attr(report, "research"),
                    "forecast_rationales": safe_report_attr(report, "forecast_rationales"),
                    "explanation": getattr(report, "explanation", None),
                }
                significant_changes.append(change)
                significant_changes_by_account[account_name].append(change)

    new_state = {"version": 1, "generated_at": now_iso, "questions": new_state_questions}
    state_path.write_text(
        json.dumps(new_state, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    digest_path = out_dir / f"digest_{now_iso[:10]}.md"
    changes_path = out_dir / "changes.md"
    failures_path = out_dir / "failures.json"

    def fmt_pred(pred: object | None, qtype: str) -> str:
        return _format_prediction_for_markdown(pred=pred, question_type=qtype)

    lines: list[str] = []
    lines.append(f"# Metaculus digest ({now_iso})")
    lines.append("")
    lines.append(f"- Compare mode: `{compare_mode}`")
    lines.append(
        f"- Max questions/tournament: `{max_questions_per_tournament if max_questions_per_tournament is not None else 'all'}`"
    )
    lines.append(f"- Track account enabled: `{bool(track_token)}`")
    lines.append("")
    lines.append("## Tournaments")
    for tid in tournaments:
        lines.append(f"- {tid}")
    lines.append("")

    if compare_mode == "state":
        lines.append(f"## Significant changes ({len(significant_changes)})")
        if not significant_changes:
            lines.append("- (none)")
        else:
            for ch in significant_changes:
                url = ch.get("page_url") or ch.get("key")
                qtype = ch.get("question_type") or "unknown"
                lines.append(f"### {ch['question_text']}")
                lines.append(f"- Tournament: {ch['tournament']}")
                lines.append(f"- URL: {url}")
                if ch.get("close_time"):
                    lines.append(f"- Close time (UTC): {ch['close_time']}")
                lines.append(f"- Type: {qtype}")
                lines.append(f"- Reason: {ch['reason']}")
                lines.append(f"- Old: {fmt_pred(ch.get('old_prediction'), qtype)}")
                lines.append(f"- New: {fmt_pred(ch.get('new_prediction'), qtype)}")
                lines.append("")
                lines.append("**Readable prediction**")
                lines.append(ch.get("readable_prediction") or "")
                lines.append("")
                if ch.get("summary"):
                    lines.append("**Report summary**")
                    lines.append(str(ch["summary"]))
                    lines.append("")
                if ch.get("research"):
                    lines.append("**Report research**")
                    lines.append(str(ch["research"]))
                    lines.append("")
                if ch.get("forecast_rationales"):
                    lines.append("**Report forecast**")
                    lines.append(str(ch["forecast_rationales"]))
                    lines.append("")
    else:
        bot_changes = significant_changes_by_account.get("bot") or []
        track_changes = significant_changes_by_account.get("track") or []

        lines.append(f"## Significant diffs vs bot account ({len(bot_changes)})")
        if not bot_changes:
            lines.append("- (none)")
        else:
            for ch in bot_changes:
                url = ch.get("page_url") or ch.get("key")
                qtype = ch.get("question_type") or "unknown"
                lines.append(f"### {ch['question_text']}")
                lines.append(f"- Tournament: {ch['tournament']}")
                lines.append(f"- URL: {url}")
                if ch.get("close_time"):
                    lines.append(f"- Close time (UTC): {ch['close_time']}")
                lines.append(f"- Type: {qtype}")
                lines.append(f"- Reason: {ch['reason']}")
                lines.append(f"- Bot current: {fmt_pred(ch.get('account_prediction'), qtype)}")
                lines.append(f"- AI new: {fmt_pred(ch.get('ai_prediction'), qtype)}")
                lines.append("")
                lines.append("**Readable prediction**")
                lines.append(ch.get("readable_prediction") or "")
                lines.append("")
                if ch.get("summary"):
                    lines.append("**Report summary**")
                    lines.append(str(ch["summary"]))
                    lines.append("")
                if ch.get("research"):
                    lines.append("**Report research**")
                    lines.append(str(ch["research"]))
                    lines.append("")
                if ch.get("forecast_rationales"):
                    lines.append("**Report forecast**")
                    lines.append(str(ch["forecast_rationales"]))
                    lines.append("")

        if track_token:
            lines.append(f"## Significant diffs vs tracked account ({len(track_changes)})")
            if not track_changes:
                lines.append("- (none)")
            else:
                for ch in track_changes:
                    url = ch.get("page_url") or ch.get("key")
                    qtype = ch.get("question_type") or "unknown"
                    lines.append(f"### {ch['question_text']}")
                    lines.append(f"- Tournament: {ch['tournament']}")
                    lines.append(f"- URL: {url}")
                    if ch.get("close_time"):
                        lines.append(f"- Close time (UTC): {ch['close_time']}")
                    lines.append(f"- Type: {qtype}")
                    lines.append(f"- Reason: {ch['reason']}")
                    lines.append(f"- Tracked current: {fmt_pred(ch.get('account_prediction'), qtype)}")
                    lines.append(f"- AI new: {fmt_pred(ch.get('ai_prediction'), qtype)}")
                    lines.append("")
                    lines.append("**Readable prediction**")
                    lines.append(ch.get("readable_prediction") or "")
                    lines.append("")
                    if ch.get("summary"):
                        lines.append("**Report summary**")
                        lines.append(str(ch["summary"]))
                        lines.append("")
                    if ch.get("research"):
                        lines.append("**Report research**")
                        lines.append(str(ch["research"]))
                        lines.append("")
                    if ch.get("forecast_rationales"):
                        lines.append("**Report forecast**")
                        lines.append(str(ch["forecast_rationales"]))
                        lines.append("")

        unforecasted_bot = unforecasted_by_account.get("bot") or []
        unforecasted_track = unforecasted_by_account.get("track") or []
        lines.append(f"## Unforecasted by bot account ({len(unforecasted_bot)})")
        if not unforecasted_bot:
            lines.append("- (none)")
        else:
            for item in unforecasted_bot[:50]:
                url = item.get("page_url") or item.get("key")
                lines.append(f"- {item.get('tournament')}: {item.get('question_text')} ({url})")
            if len(unforecasted_bot) > 50:
                lines.append(f"- ... and {len(unforecasted_bot) - 50} more")

        if track_token:
            lines.append(f"## Unforecasted by tracked account ({len(unforecasted_track)})")
            if not unforecasted_track:
                lines.append("- (none)")
            else:
                for item in unforecasted_track[:50]:
                    url = item.get("page_url") or item.get("key")
                    lines.append(
                        f"- {item.get('tournament')}: {item.get('question_text')} ({url})"
                    )
                if len(unforecasted_track) > 50:
                    lines.append(f"- ... and {len(unforecasted_track) - 50} more")

    lines.append("")
    lines.append(f"## Failures ({len(failures)})")
    if not failures:
        lines.append("- (none)")
    else:
        lines.append(f"- See `failures.json` in the digest artifact for full details.")
        for failure in failures[:20]:
            tournament = failure.get("tournament", "(unknown tournament)")
            error = failure.get("error", "(unknown error)")
            lines.append(f"- {tournament}: {str(error)[:300]}")
        if len(failures) > 20:
            lines.append(f"- ... and {len(failures) - 20} more")

    digest_path.write_text("\n".join(lines), encoding="utf-8")
    changes_path.write_text("\n".join(lines), encoding="utf-8")
    failures_path.write_text(
        json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if significant_changes or failures:
        if compare_mode == "account":
            bot_count = len(significant_changes_by_account.get("bot") or [])
            track_count = len(significant_changes_by_account.get("track") or [])
            summary = (
                f"Metaculus digest ({now_iso[:10]}): bot Δ {bot_count}"
                + (f", tracked Δ {track_count}" if track_token else "")
            )
        else:
            summary = (
                f"Metaculus digest: {len(significant_changes)} significant change(s) ({now_iso[:10]})"
            )

        if failures and not significant_changes:
            summary += f" (failures: {len(failures)})"
        elif failures:
            summary += f" (failures: {len(failures)})"

        message_lines = [summary]
        for ch in significant_changes[:10]:
            url = ch.get("page_url") or ch.get("key")
            prefix = ""
            if compare_mode == "account":
                prefix = f"[{ch.get('account')}] "
            message_lines.append(
                f"- {prefix}{ch['tournament']}: {ch['question_text']} ({url})"
            )
        if len(significant_changes) > 10:
            message_lines.append(f"... and {len(significant_changes) - 10} more")

        if failures and not significant_changes:
            message_lines.append("")
            message_lines.append("Failures (first 3):")
            for failure in failures[:3]:
                tournament = failure.get("tournament", "(unknown tournament)")
                error = failure.get("error", "(unknown error)")
                message_lines.append(f"- {tournament}: {str(error)[:200]}")
        matrix_send_message("\n".join(message_lines))
