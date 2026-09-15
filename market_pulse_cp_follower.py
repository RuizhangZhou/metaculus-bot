#!/usr/bin/env python3
"""Refresh a Market Pulse forecast from the visible Community Prediction daily.

This deliberately contains no forecasting or research fallback. A question is
eligible only after ``cp_reveal_time`` and before ``spot_scoring_time``. Every
daily run resubmits the then-current Community Prediction so the standing
forecast follows later community movement. No local queue database is needed.

The official API exposes Community Predictions on only a limited set of
questions for ordinary/restricted tokens.  If an eligible question has no
aggregation data, the command fails loudly and submits nothing invented.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any


API_BASE = "https://www.metaculus.com/api"
MARKET_PULSE_SLUG = re.compile(r"^market-pulse-\d{2}q[1-4]$", re.IGNORECASE)


class APIError(RuntimeError):
    pass


@dataclass(frozen=True)
class Candidate:
    question_id: int
    post_id: int
    title: str
    reveal_time: datetime
    score_time: datetime
    payload: dict[str, Any]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def parse_datetime(value: object) -> datetime | None:
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
            timezone.utc
        )
    except ValueError:
        return None


def _float_list(value: object) -> list[float] | None:
    if not isinstance(value, list):
        return None
    if not all(isinstance(item, (int, float)) for item in value):
        return None
    return [float(item) for item in value]


def _options(question: dict) -> list[str]:
    result: list[str] = []
    for option in question.get("options") or []:
        if isinstance(option, str):
            result.append(option)
        elif isinstance(option, dict):
            label = option.get("name") or option.get("label") or option.get("title")
            if isinstance(label, str):
                result.append(label)
    return result


def latest_community_aggregation(question: dict) -> dict | None:
    aggregations = question.get("aggregations")
    if not isinstance(aggregations, dict):
        return None
    for method in ("recency_weighted", "unweighted"):
        block = aggregations.get(method)
        latest = block.get("latest") if isinstance(block, dict) else None
        if isinstance(latest, dict):
            return latest
    return None


def build_community_payload(question: dict) -> dict[str, Any] | None:
    """Turn the latest aggregate into the API forecast payload, losslessly."""
    latest = latest_community_aggregation(question)
    if latest is None:
        return None

    question_type = question.get("type")
    values = _float_list(latest.get("forecast_values"))

    if question_type == "binary":
        probability: float | None = None
        for key in ("centers", "means"):
            candidate = _float_list(latest.get(key))
            if candidate and len(candidate) == 1:
                probability = candidate[0]
                break
        if probability is None and values and len(values) >= 2:
            probability = values[1]
        if probability is None:
            return None
        return {
            "probability_yes": min(0.999, max(0.001, probability)),
            "probability_yes_per_category": None,
            "continuous_cdf": None,
        }

    if question_type == "multiple_choice":
        per_category = latest.get("probability_yes_per_category")
        if isinstance(per_category, dict) and per_category:
            mapped = {
                str(key): float(value)
                for key, value in per_category.items()
                if isinstance(key, str) and isinstance(value, (int, float))
            }
        else:
            labels = _options(question)
            mapped = (
                {label: value for label, value in zip(labels, values)}
                if values and len(labels) == len(values)
                else {}
            )
        if not mapped:
            return None
        return {
            "probability_yes": None,
            "probability_yes_per_category": mapped,
            "continuous_cdf": None,
        }

    if question_type in {"numeric", "date", "discrete"}:
        if values is None:
            values = _float_list(latest.get("continuous_cdf"))
        # Metaculus continuous forecasts use the exact 201-point CDF exposed by
        # the aggregate.  Refuse interpolation here: this command only copies.
        if (
            values is None
            or len(values) != 201
            or not all(0.0 <= value <= 1.0 for value in values)
            or not all(left <= right for left, right in zip(values, values[1:]))
        ):
            return None
        return {
            "probability_yes": None,
            "probability_yes_per_category": None,
            "continuous_cdf": values,
        }

    return None


def candidate_for_question(
    question: dict,
    *,
    post_id: int,
    now: datetime,
    grace_after_reveal: timedelta,
) -> tuple[Candidate | None, str]:
    if question.get("status") != "open":
        return None, "not_open"

    question_id = question.get("id")
    if not isinstance(question_id, int):
        return None, "missing_id"

    reveal = parse_datetime(question.get("cp_reveal_time"))
    score = parse_datetime(question.get("spot_scoring_time")) or parse_datetime(
        question.get("scheduled_close_time")
    )
    if reveal is None or score is None:
        return None, "missing_timing"
    if score <= reveal + grace_after_reveal:
        return None, "no_copy_window"
    if now < reveal + grace_after_reveal:
        return None, "waiting_for_reveal"
    if now >= score:
        return None, "score_window_passed"

    payload = build_community_payload(question)
    if payload is None:
        return None, "community_prediction_unavailable"
    title = question.get("title")
    return (
        Candidate(
            question_id=question_id,
            post_id=post_id,
            title=title if isinstance(title, str) else f"Question {question_id}",
            reveal_time=reveal,
            score_time=score,
            payload=payload,
        ),
        "ready",
    )


class MetaculusAPI:
    def __init__(self, token: str, *, timeout: int = 60) -> None:
        if not token:
            raise ValueError("METACULUS_TOKEN is required")
        self.token = token
        self.timeout = timeout

    def request(
        self, path: str, *, method: str = "GET", payload: object | None = None
    ) -> object:
        data = None
        headers = {
            "Authorization": f"Token {self.token}",
            "Accept": "application/json",
            "User-Agent": "metaculus-market-pulse-cp-follower/1.0",
        }
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            f"{API_BASE}{path}", data=data, method=method, headers=headers
        )

        last_error: Exception | None = None
        for attempt in range(4):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw = response.read()
                    return json.loads(raw) if raw else None
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code not in {429, 500, 502, 503, 504}:
                    break
            except (OSError, TimeoutError) as exc:
                last_error = exc
            time.sleep(2 * (attempt + 1))
        raise APIError(f"Metaculus API {method} {path} failed: {last_error!r}")

    def current_user(self) -> dict:
        result = self.request("/users/me/")
        if not isinstance(result, dict):
            raise APIError("Metaculus /users/me/ returned an unexpected response")
        return result

    def tournament_exists(self, slug: str) -> bool:
        result = self.request("/projects/tournaments/")
        if isinstance(result, list):
            tournaments = result
        elif isinstance(result, dict):
            tournaments = result.get("results", [])
        else:
            tournaments = []
        return any(
            isinstance(item, dict) and item.get("slug") == slug
            for item in tournaments
        )

    def tournament_questions(self, slug: str) -> list[tuple[int, dict]]:
        query = urllib.parse.urlencode({"tournaments": slug, "limit": 100})
        listing = self.request(f"/posts/?{query}")
        if not isinstance(listing, dict):
            raise APIError("Metaculus posts listing returned an unexpected response")

        questions: list[tuple[int, dict]] = []
        for item in listing.get("results") or []:
            post_id = item.get("id") if isinstance(item, dict) else None
            if not isinstance(post_id, int):
                continue
            post = self.request(f"/posts/{post_id}/")
            if not isinstance(post, dict):
                continue
            group = post.get("group_of_questions")
            single = post.get("question")
            children = (
                group.get("questions")
                if isinstance(group, dict)
                else ([single] if isinstance(single, dict) else [])
            )
            for question in children or []:
                if isinstance(question, dict):
                    questions.append((post_id, question))
        return questions

    def submit(self, candidate: Candidate) -> None:
        body = [
            {
                "question": candidate.question_id,
                "source": "api",
                **candidate.payload,
            }
        ]
        self.request("/questions/forecast/", method="POST", payload=body)


def _write_step_summary(
    *, tournament: str, username: str, submit: bool, counts: dict[str, int]
) -> None:
    path = os.getenv("GITHUB_STEP_SUMMARY", "").strip()
    if not path:
        return
    lines = [
        "## Market Pulse Community Prediction follower",
        "",
        f"- Tournament: `{tournament}`",
        f"- Account: `{username}`",
        f"- Mode: `{'submit' if submit else 'dry-run'}`",
        "",
        "| Result | Questions |",
        "|---|---:|",
    ]
    lines.extend(f"| {key} | {value} |" for key, value in sorted(counts.items()))
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def run(
    *,
    api: MetaculusAPI,
    tournament: str,
    expected_username: str,
    submit: bool,
    now: datetime,
    grace_minutes: int,
) -> int:
    tournament = tournament.strip().lower().rstrip("/").split("/")[-1]
    if not MARKET_PULSE_SLUG.fullmatch(tournament):
        raise ValueError(
            "Refusing non-Market-Pulse target; expected a slug like market-pulse-26q4"
        )

    user = api.current_user()
    username = str(user.get("username") or "unknown")
    if submit and not expected_username:
        raise RuntimeError(
            "MARKET_PULSE_EXPECTED_USERNAME is required in submit mode"
        )
    if expected_username and username.casefold() != expected_username.casefold():
        raise RuntimeError(
            f"Refusing to use account {username!r}; expected {expected_username!r}"
        )
    print(
        f"account={username} is_bot={bool(user.get('is_bot'))} "
        f"api_access_tier={user.get('api_access_tier') or 'unknown'}"
    )

    if not api.tournament_exists(tournament):
        print(f"waiting: tournament {tournament!r} is not published in the API yet")
        _write_step_summary(
            tournament=tournament,
            username=username,
            submit=submit,
            counts={"waiting_for_tournament": 1},
        )
        return 0

    questions = api.tournament_questions(tournament)
    counts: dict[str, int] = {"questions_seen": len(questions)}
    candidates: list[Candidate] = []
    grace = timedelta(minutes=grace_minutes)
    for post_id, question in questions:
        candidate, reason = candidate_for_question(
            question,
            post_id=post_id,
            now=now,
            grace_after_reveal=grace,
        )
        counts[reason] = counts.get(reason, 0) + 1
        if candidate is not None:
            candidates.append(candidate)

    submitted = 0
    for candidate in candidates:
        url = f"https://www.metaculus.com/questions/{candidate.post_id}/"
        if submit:
            api.submit(candidate)
            submitted += 1
            print(
                f"submitted qid={candidate.question_id} score_at="
                f"{candidate.score_time.isoformat()} {url}"
            )
        else:
            print(
                f"dry-run ready qid={candidate.question_id} score_at="
                f"{candidate.score_time.isoformat()} {url}"
            )
    if submitted:
        counts["submitted"] = submitted

    print("summary=" + json.dumps(counts, sort_keys=True))
    _write_step_summary(
        tournament=tournament,
        username=username,
        submit=submit,
        counts=counts,
    )

    unavailable = counts.get("community_prediction_unavailable", 0)
    if unavailable:
        print(
            "::error::Eligible Market Pulse question(s) have no Community "
            "Prediction in this API response. Use a token with CP data access "
            "or submit manually; no fallback forecast was made.",
            file=sys.stderr,
        )
        return 2
    if counts.get("no_copy_window", 0):
        print(
            "::warning::At least one question has no safe interval between CP "
            "reveal and spot scoring; copying it after reveal is impossible.",
            file=sys.stderr,
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tournament",
        default=os.getenv("MARKET_PULSE_CP_TOURNAMENT", "market-pulse-26q4"),
    )
    parser.add_argument(
        "--expected-username",
        default=os.getenv("MARKET_PULSE_EXPECTED_USERNAME", ""),
        help="Refuse to run if METACULUS_TOKEN belongs to another account.",
    )
    parser.add_argument("--submit", action="store_true")
    parser.add_argument(
        "--grace-minutes",
        type=int,
        default=int(os.getenv("MARKET_PULSE_CP_GRACE_MINUTES", "5")),
    )
    args = parser.parse_args()
    if args.grace_minutes < 0:
        parser.error("grace-minutes must be non-negative")

    token = os.getenv("METACULUS_TOKEN", "").strip()
    try:
        return run(
            api=MetaculusAPI(token),
            tournament=args.tournament,
            expected_username=args.expected_username.strip(),
            submit=args.submit,
            now=utcnow(),
            grace_minutes=args.grace_minutes,
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"::error::{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
