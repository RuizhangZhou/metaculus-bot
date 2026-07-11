from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

from forecasting_tools import MetaculusClient, MetaculusQuestion

from tournament_update import (
    _extract_cp_latest,
    _extract_float_list,
    _extract_options,
    _get_full_question_json,
    _question_json_from_question,
    _question_type_string,
    _select_aggregation_block,
)

logger = logging.getLogger(__name__)

PredictionKind = Literal["binary", "multiple_choice", "numeric"]


@dataclass(frozen=True)
class CommunityPredictionPayload:
    question_id: int
    question_type: str
    kind: PredictionKind
    value: float | dict[str, float] | list[float]


def _clip_binary_prediction(value: float) -> float:
    return min(0.999, max(0.001, float(value)))


def _latest_aggregation_item(question_json: dict) -> dict | None:
    aggregation_block = _select_aggregation_block(question_json)
    if not isinstance(aggregation_block, dict):
        return None
    latest = aggregation_block.get("latest")
    return latest if isinstance(latest, dict) else None


def _community_numeric_cdf(question_json: dict) -> list[float] | None:
    latest = _latest_aggregation_item(question_json)
    if not isinstance(latest, dict):
        return None

    values = _extract_float_list(latest.get("forecast_values"))
    if values is None:
        values = _extract_float_list(latest.get("continuous_cdf"))
    if not values or len(values) < 2:
        return None
    if not all(0.0 <= x <= 1.0 for x in values):
        return None
    if not all(a <= b for a, b in zip(values, values[1:])):
        return None
    return values


def build_community_prediction_payload(
    question: MetaculusQuestion, *, question_json: dict | None = None
) -> CommunityPredictionPayload | None:
    question_id = getattr(question, "id_of_question", None)
    if not isinstance(question_id, int):
        return None

    question_type = _question_type_string(question)
    question_json = (
        question_json
        if isinstance(question_json, dict)
        else _question_json_from_question(question)
    )

    if question_type == "binary":
        cp = getattr(question, "community_prediction_at_access_time", None)
        if not isinstance(cp, (int, float)) and isinstance(question_json, dict):
            cp = _extract_cp_latest(
                question_json=question_json, question_type=question_type
            )
        if not isinstance(cp, (int, float)):
            return None
        return CommunityPredictionPayload(
            question_id=question_id,
            question_type=question_type,
            kind="binary",
            value=_clip_binary_prediction(float(cp)),
        )

    if not isinstance(question_json, dict):
        return None

    if question_type == "multiple_choice":
        cp = _extract_cp_latest(question_json=question_json, question_type=question_type)
        if isinstance(cp, dict):
            mapped = {
                str(k): float(v)
                for k, v in cp.items()
                if isinstance(k, str) and isinstance(v, (int, float))
            }
            if mapped:
                return CommunityPredictionPayload(
                    question_id=question_id,
                    question_type=question_type,
                    kind="multiple_choice",
                    value=mapped,
                )

        latest = _latest_aggregation_item(question_json)
        values = _extract_float_list((latest or {}).get("forecast_values"))
        if values is None:
            values = _extract_float_list((latest or {}).get("centers"))
        options = _extract_options(question_json)
        if values and options and len(values) == len(options):
            return CommunityPredictionPayload(
                question_id=question_id,
                question_type=question_type,
                kind="multiple_choice",
                value={option: float(prob) for option, prob in zip(options, values)},
            )
        return None

    if question_type in {"numeric", "date", "discrete"}:
        values = _community_numeric_cdf(question_json)
        if values:
            return CommunityPredictionPayload(
                question_id=question_id,
                question_type=question_type,
                kind="numeric",
                value=values,
            )
        return None

    return None


def _question_json_for_sync(
    *,
    client: MetaculusClient,
    question: MetaculusQuestion,
    full_question_cache: dict[int, dict[int, dict]],
) -> dict | None:
    question_json = _question_json_from_question(question)
    if isinstance(question_json, dict):
        return question_json

    question_id = getattr(question, "id_of_question", None)
    post_id = getattr(question, "id_of_post", None)
    if not isinstance(question_id, int) or not isinstance(post_id, int):
        return None
    return _get_full_question_json(
        client=client,
        post_id=post_id,
        question_id=question_id,
        cache=full_question_cache,
    )


def post_community_prediction_payload(
    *, client: MetaculusClient, payload: CommunityPredictionPayload
) -> None:
    if payload.kind == "binary":
        client.post_binary_question_prediction(payload.question_id, float(payload.value))
        return
    if payload.kind == "multiple_choice":
        if not isinstance(payload.value, dict):
            raise TypeError("multiple_choice community payload must be a dict")
        client.post_multiple_choice_question_prediction(payload.question_id, payload.value)
        return
    if payload.kind == "numeric":
        if not isinstance(payload.value, list):
            raise TypeError("numeric community payload must be a CDF list")
        client.post_numeric_question_prediction(payload.question_id, payload.value)
        return
    raise ValueError(f"Unsupported community prediction payload kind: {payload.kind}")


def sync_questions_to_community_predictions(
    *,
    client: MetaculusClient,
    questions: list[MetaculusQuestion],
    publish: bool,
) -> tuple[list[MetaculusQuestion], dict[str, int]]:
    full_question_cache: dict[int, dict[int, dict]] = {}
    remaining: list[MetaculusQuestion] = []
    counts = {
        "checked": 0,
        "synced": 0,
        "dry_run_synced": 0,
        "missing_community_prediction": 0,
        "failed": 0,
    }

    for question in questions:
        counts["checked"] += 1
        question_json = _question_json_for_sync(
            client=client,
            question=question,
            full_question_cache=full_question_cache,
        )
        payload = build_community_prediction_payload(
            question, question_json=question_json
        )
        if payload is None:
            counts["missing_community_prediction"] += 1
            remaining.append(question)
            continue

        url = getattr(question, "page_url", None) or f"question:{payload.question_id}"
        if not publish:
            logger.info(
                "Dry-run community prediction sync skipped pipeline for %s (%s)",
                url,
                payload.question_type,
            )
            counts["dry_run_synced"] += 1
            continue

        try:
            post_community_prediction_payload(client=client, payload=payload)
            counts["synced"] += 1
            logger.info(
                "Synced %s to community prediction for %s",
                payload.question_type,
                url,
            )
        except Exception:
            counts["failed"] += 1
            logger.exception("Failed to sync community prediction for %s", url)
            remaining.append(question)

    return remaining, counts
