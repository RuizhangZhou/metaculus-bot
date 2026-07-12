from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Callable, Iterable


_CONTEXT_METADATA_KEY = "financial_market_context"
_METRICS_METADATA_KEY = "financial_market_metrics"

_SNAPSHOT_HEADER_RE = re.compile(r"^- .+\((?P<symbol>[^();]+);\s*(?P<source>[^)]+)\):")
_FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


@dataclass(frozen=True)
class FinancialUpdateGatingOptions:
    enabled: bool
    state_path: Path
    log_path: Path
    baseline_delta_threshold: float = 0.03
    price_sigma_threshold: float = 0.50
    always_forecast_after_hours: float = 72.0
    always_forecast_within_days: float = 2.0


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    lowered = raw.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except Exception:
        return default


def financial_update_gating_options_from_env() -> FinancialUpdateGatingOptions:
    return FinancialUpdateGatingOptions(
        enabled=_env_bool("BOT_ENABLE_FINANCIAL_UPDATE_GATING", False),
        state_path=Path(
            os.getenv(
                "BOT_FINANCIAL_UPDATE_STATE_PATH",
                ".state/financial_update_state.json",
            )
        ),
        log_path=Path(
            os.getenv(
                "BOT_FINANCIAL_STRUCTURED_LOG_PATH",
                ".state/financial_predictions.jsonl",
            )
        ),
        baseline_delta_threshold=_env_float(
            "BOT_FINANCIAL_UPDATE_GATE_BASELINE_DELTA", 0.03
        ),
        price_sigma_threshold=_env_float(
            "BOT_FINANCIAL_UPDATE_GATE_PRICE_SIGMA", 0.50
        ),
        always_forecast_after_hours=_env_float(
            "BOT_FINANCIAL_UPDATE_GATE_ALWAYS_AFTER_HOURS", 72.0
        ),
        always_forecast_within_days=_env_float(
            "BOT_FINANCIAL_UPDATE_GATE_ALWAYS_WITHIN_DAYS", 2.0
        ),
    )


def question_key(question: Any) -> str:
    parts = [
        str(getattr(question, "id_of_post", "") or ""),
        str(getattr(question, "id_of_question", "") or ""),
        str(getattr(question, "conditional_type", "") or ""),
        str(getattr(question, "group_question_option", "") or ""),
    ]
    if any(parts):
        return "|".join(parts)
    return str(getattr(question, "page_url", "") or getattr(question, "question_text", ""))


def _ensure_metadata(question: Any) -> dict[str, Any]:
    metadata = getattr(question, "custom_metadata", None)
    if not isinstance(metadata, dict):
        metadata = {}
        try:
            setattr(question, "custom_metadata", metadata)
        except Exception:
            return {}
    return metadata


def get_cached_financial_context(question: Any) -> str:
    metadata = getattr(question, "custom_metadata", None)
    if not isinstance(metadata, dict):
        return ""
    value = metadata.get(_CONTEXT_METADATA_KEY)
    return value if isinstance(value, str) else ""


def set_cached_financial_context(
    question: Any, context: str, metrics: dict[str, Any] | None = None
) -> None:
    if not context:
        return
    metadata = _ensure_metadata(question)
    if not isinstance(metadata, dict):
        return
    metadata[_CONTEXT_METADATA_KEY] = context
    if metrics:
        metadata[_METRICS_METADATA_KEY] = metrics


def get_cached_financial_metrics(question: Any) -> dict[str, Any] | None:
    metadata = getattr(question, "custom_metadata", None)
    if not isinstance(metadata, dict):
        return None
    value = metadata.get(_METRICS_METADATA_KEY)
    return value if isinstance(value, dict) else None


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        f = float(value.replace(",", ""))
    except Exception:
        return None
    if not math.isfinite(f):
        return None
    return f


def _search_float(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        return None
    return _parse_float(match.group(1))


def extract_financial_metrics_from_context(context: str) -> dict[str, Any] | None:
    if "Financial question spec:" not in (context or ""):
        return None
    text = context or ""

    target_kind_match = re.search(r"^- target_kind:\s*(.+)$", text, re.MULTILINE)
    threshold_match = re.search(
        rf"^- threshold:\s*(?P<direction>\S+)\s+(?P<value>{_FLOAT_RE})",
        text,
        re.MULTILINE,
    )
    target_dt_match = re.search(
        r"^- target_datetime_utc_or_site:\s*(.+)$", text, re.MULTILINE
    )
    broad_news_match = re.search(r"^- Broad news default:\s*(.+)$", text, re.MULTILINE)

    snapshot_symbol = None
    snapshot_source = None
    for line in text.splitlines():
        match = _SNAPSHOT_HEADER_RE.match(line.strip())
        if match:
            snapshot_symbol = match.group("symbol").strip()
            snapshot_source = match.group("source").strip()
            break

    latest_quote = _search_float(rf"^\s+- latest_quote:\s*({_FLOAT_RE})", text)
    latest_daily_close = _search_float(
        rf"^\s+- latest_daily_close:\s*\d{{4}}-\d{{2}}-\d{{2}}:\s*({_FLOAT_RE})",
        text,
    )
    daily_close_date_match = re.search(
        r"^\s+- latest_daily_close:\s*(\d{4}-\d{2}-\d{2}):",
        text,
        re.MULTILINE,
    )
    quote_time_match = re.search(r"^\s+- quote_time_utc:\s*(.+)$", text, re.MULTILINE)
    source_match = re.search(r"^\s+- source:\s*(https?://\S+)", text, re.MULTILINE)

    vol_match = re.search(
        r"30d_daily=([0-9.]+)%, 30d_annualized=([0-9.]+)%, "
        r"90d_daily=([0-9.]+)%, 90d_annualized=([0-9.]+)%",
        text,
    )
    daily_vol_30 = _parse_float(vol_match.group(1)) / 100.0 if vol_match else None
    annual_vol_30 = _parse_float(vol_match.group(2)) / 100.0 if vol_match else None
    daily_vol_90 = _parse_float(vol_match.group(3)) / 100.0 if vol_match else None
    annual_vol_90 = _parse_float(vol_match.group(4)) / 100.0 if vol_match else None

    baseline_probability = _search_float(
        r"baseline_probability_for_prompt:\s*([0-9.]+)%", text
    )
    if baseline_probability is not None:
        baseline_probability /= 100.0
    current_price_anchor = _search_float(
        rf"current_price_anchor:\s*({_FLOAT_RE})", text
    )
    distance_to_threshold = _search_float(
        rf"distance_to_threshold:\s*({_FLOAT_RE})", text
    )
    horizon_days = _search_float(rf"horizon_days:\s*({_FLOAT_RE})", text)

    current_price = current_price_anchor or latest_daily_close or latest_quote
    if snapshot_symbol is None and latest_quote is None and baseline_probability is None:
        return None

    metrics: dict[str, Any] = {
        "symbol": snapshot_symbol,
        "source": snapshot_source,
        "target_kind": target_kind_match.group(1).strip()
        if target_kind_match
        else None,
        "threshold_direction": threshold_match.group("direction")
        if threshold_match
        else None,
        "threshold": _parse_float(threshold_match.group("value"))
        if threshold_match
        else None,
        "target_datetime": target_dt_match.group(1).strip()
        if target_dt_match
        else None,
        "broad_news_default": broad_news_match.group(1).strip()
        if broad_news_match
        else None,
        "latest_quote": latest_quote,
        "quote_time_utc": quote_time_match.group(1).strip()
        if quote_time_match
        else None,
        "latest_daily_close": latest_daily_close,
        "latest_daily_close_date": daily_close_date_match.group(1)
        if daily_close_date_match
        else None,
        "current_price_anchor": current_price,
        "daily_vol_30": daily_vol_30,
        "annual_vol_30": annual_vol_30,
        "daily_vol_90": daily_vol_90,
        "annual_vol_90": annual_vol_90,
        "baseline_probability": baseline_probability,
        "distance_to_threshold": distance_to_threshold,
        "horizon_days": horizon_days,
        "source_url": source_match.group(1) if source_match else None,
    }
    return metrics


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "questions": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"version": 1, "questions": {}}
    if not isinstance(data, dict):
        return {"version": 1, "questions": {}}
    questions = data.get("questions")
    if not isinstance(questions, dict):
        data["questions"] = {}
    data.setdefault("version", 1)
    return data


def _save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _jsonable(model_dump())
    as_dict = getattr(value, "dict", None)
    if callable(as_dict):
        return _jsonable(as_dict())
    return str(value)


def _append_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_jsonable(event), ensure_ascii=False, sort_keys=True))
        f.write("\n")


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _hours_since(value: str | None, now: datetime) -> float | None:
    parsed = _parse_iso(value)
    if parsed is None:
        return None
    return (now - parsed).total_seconds() / 3600.0


def _price_sigma_move(
    current: dict[str, Any], previous: dict[str, Any]
) -> float | None:
    current_price = current.get("current_price_anchor")
    previous_price = previous.get("current_price_anchor")
    if not isinstance(current_price, (int, float)) or not isinstance(
        previous_price, (int, float)
    ):
        return None
    if current_price <= 0 or previous_price <= 0:
        return None
    vols = [
        current.get("daily_vol_30"),
        previous.get("daily_vol_30"),
        current.get("daily_vol_90"),
        previous.get("daily_vol_90"),
    ]
    valid_vols = [float(v) for v in vols if isinstance(v, (int, float)) and v > 0]
    if not valid_vols:
        return None
    return abs(math.log(float(current_price) / float(previous_price))) / max(valid_vols)


def _gating_decision(
    *,
    current_metrics: dict[str, Any],
    previous_state: dict[str, Any] | None,
    options: FinancialUpdateGatingOptions,
    now: datetime,
) -> tuple[bool, str, dict[str, Any]]:
    if not isinstance(previous_state, dict):
        return True, "no_previous_financial_state", {}

    previous_metrics = previous_state.get("last_forecast_metrics")
    if not isinstance(previous_metrics, dict):
        return True, "missing_previous_forecast_metrics", {}

    age_hours = _hours_since(previous_state.get("last_forecast_at"), now)
    if age_hours is None:
        return True, "missing_last_forecast_at", {}
    if options.always_forecast_after_hours > 0 and age_hours >= options.always_forecast_after_hours:
        return True, f"max_age_hours={age_hours:.1f}", {"age_hours": age_hours}

    horizon_days = current_metrics.get("horizon_days")
    if (
        isinstance(horizon_days, (int, float))
        and options.always_forecast_within_days > 0
        and float(horizon_days) <= options.always_forecast_within_days
    ):
        return (
            True,
            f"near_resolution_days={float(horizon_days):.2f}",
            {"horizon_days": float(horizon_days)},
        )

    current_baseline = current_metrics.get("baseline_probability")
    previous_baseline = previous_metrics.get("baseline_probability")
    if not isinstance(current_baseline, (int, float)) or not isinstance(
        previous_baseline, (int, float)
    ):
        return True, "missing_baseline_probability", {}
    baseline_delta = abs(float(current_baseline) - float(previous_baseline))
    if baseline_delta >= options.baseline_delta_threshold:
        return (
            True,
            f"baseline_delta={baseline_delta:.3f}",
            {"baseline_delta": baseline_delta, "age_hours": age_hours},
        )

    sigma_move = _price_sigma_move(current_metrics, previous_metrics)
    if sigma_move is not None and sigma_move >= options.price_sigma_threshold:
        return (
            True,
            f"price_sigma_move={sigma_move:.3f}",
            {
                "price_sigma_move": sigma_move,
                "baseline_delta": baseline_delta,
                "age_hours": age_hours,
            },
        )

    details = {"baseline_delta": baseline_delta, "age_hours": age_hours}
    if sigma_move is not None:
        details["price_sigma_move"] = sigma_move
    return False, "financial_unchanged", details


def filter_questions_for_financial_update_gating(
    *,
    questions: Iterable[Any],
    options: FinancialUpdateGatingOptions,
    build_context: Callable[[Any], str],
    tournament_id: str,
    now: datetime | None = None,
) -> tuple[list[Any], dict[str, int]]:
    if not options.enabled:
        return list(questions), {"disabled": 1}

    now = now or datetime.now(timezone.utc)
    state = _load_state(options.state_path)
    state_questions = state.setdefault("questions", {})
    if not isinstance(state_questions, dict):
        state_questions = {}
        state["questions"] = state_questions

    counts = {
        "checked": 0,
        "financial": 0,
        "queued": 0,
        "skipped": 0,
        "non_financial": 0,
        "fetch_failed": 0,
    }
    queued: list[Any] = []

    for question in questions:
        counts["checked"] += 1
        key = question_key(question)
        context = get_cached_financial_context(question)
        if not context:
            try:
                context = build_context(question)
            except Exception as exc:
                counts["fetch_failed"] += 1
                queued.append(question)
                _append_event(
                    options.log_path,
                    {
                        "event_type": "financial_gating_error",
                        "generated_at": now.isoformat(),
                        "tournament": tournament_id,
                        "question_key": key,
                        "page_url": getattr(question, "page_url", None),
                        "error": exc.__class__.__name__,
                    },
                )
                continue

        metrics = extract_financial_metrics_from_context(context)
        if not metrics:
            counts["non_financial"] += 1
            queued.append(question)
            continue

        counts["financial"] += 1
        set_cached_financial_context(question, context, metrics)
        should_forecast, reason, details = _gating_decision(
            current_metrics=metrics,
            previous_state=state_questions.get(key)
            if isinstance(state_questions, dict)
            else None,
            options=options,
            now=now,
        )

        event = {
            "event_type": "financial_gating_queued"
            if should_forecast
            else "financial_gating_skipped",
            "generated_at": now.isoformat(),
            "tournament": tournament_id,
            "question_key": key,
            "page_url": getattr(question, "page_url", None),
            "question_text": getattr(question, "question_text", None),
            "reason": reason,
            "details": details,
            "financial_metrics": metrics,
        }
        _append_event(options.log_path, event)

        existing_state = state_questions.get(key)
        if not isinstance(existing_state, dict):
            existing_state = {}
        existing_state.update(
            {
                "question_text": getattr(question, "question_text", None),
                "page_url": getattr(question, "page_url", None),
                "last_checked_at": now.isoformat(),
                "last_checked_metrics": metrics,
            }
        )
        state_questions[key] = existing_state

        if should_forecast:
            counts["queued"] += 1
            queued.append(question)
        else:
            counts["skipped"] += 1

    state["updated_at"] = now.isoformat()
    _save_state(options.state_path, state)
    return queued, counts


def record_financial_forecast_results(
    *,
    reports: Iterable[Any],
    options: FinancialUpdateGatingOptions,
    tournament_id: str,
    now: datetime | None = None,
) -> dict[str, int]:
    now = now or datetime.now(timezone.utc)
    state = _load_state(options.state_path)
    state_questions = state.setdefault("questions", {})
    if not isinstance(state_questions, dict):
        state_questions = {}
        state["questions"] = state_questions

    counts = {"checked": 0, "financial": 0, "non_financial": 0}
    for report in reports:
        if isinstance(report, BaseException):
            continue
        counts["checked"] += 1
        question = getattr(report, "question", None)
        if question is None:
            counts["non_financial"] += 1
            continue

        metrics = get_cached_financial_metrics(question)
        context = get_cached_financial_context(question)
        if not metrics and context:
            metrics = extract_financial_metrics_from_context(context)
        if not metrics:
            research = getattr(report, "research", "") or getattr(
                report, "explanation", ""
            )
            metrics = extract_financial_metrics_from_context(str(research or ""))
        if not metrics:
            counts["non_financial"] += 1
            continue

        counts["financial"] += 1
        key = question_key(question)
        prediction = _jsonable(getattr(report, "prediction", None))
        event = {
            "event_type": "financial_forecast_recorded",
            "generated_at": now.isoformat(),
            "tournament": tournament_id,
            "question_key": key,
            "page_url": getattr(question, "page_url", None),
            "question_text": getattr(question, "question_text", None),
            "question_type": getattr(question, "question_type", None),
            "group_question_option": getattr(question, "group_question_option", None),
            "prediction": prediction,
            "community_prediction_at_access_time": _jsonable(
                getattr(question, "community_prediction_at_access_time", None)
            ),
            "financial_metrics": metrics,
        }
        _append_event(options.log_path, event)

        previous = state_questions.get(key)
        if not isinstance(previous, dict):
            previous = {}
        previous.update(
            {
                "question_text": getattr(question, "question_text", None),
                "page_url": getattr(question, "page_url", None),
                "question_type": getattr(question, "question_type", None),
                "group_question_option": getattr(question, "group_question_option", None),
                "last_forecast_at": now.isoformat(),
                "last_forecast_metrics": metrics,
                "last_prediction": prediction,
                "last_community_prediction_at_access_time": _jsonable(
                    getattr(question, "community_prediction_at_access_time", None)
                ),
            }
        )
        state_questions[key] = previous

    state["updated_at"] = now.isoformat()
    _save_state(options.state_path, state)
    return counts
