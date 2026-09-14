import unittest
from datetime import datetime, timedelta, timezone

from market_pulse_cp_follower import (
    Candidate,
    already_followed_after_reveal,
    build_community_payload,
    candidate_for_question,
    run,
)


NOW = datetime(2026, 9, 20, 17, 20, tzinfo=timezone.utc)
REVEAL = datetime(2026, 9, 18, 16, 0, tzinfo=timezone.utc)
SCORE = datetime(2026, 9, 21, 3, 0, tzinfo=timezone.utc)


def numeric_question(**overrides):
    question = {
        "id": 101,
        "title": "Market value",
        "type": "numeric",
        "status": "open",
        "cp_reveal_time": REVEAL.isoformat(),
        "spot_scoring_time": SCORE.isoformat(),
        "scheduled_close_time": SCORE.isoformat(),
        "aggregations": {
            "recency_weighted": {
                "latest": {"forecast_values": [index / 200 for index in range(201)]}
            }
        },
        "my_forecasts": {"history": [], "latest": None},
    }
    question.update(overrides)
    return question


class TestCommunityPayload(unittest.TestCase):
    def test_numeric_copies_the_exact_201_point_cdf(self) -> None:
        question = numeric_question()

        payload = build_community_payload(question)

        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertEqual(
            payload["continuous_cdf"],
            question["aggregations"]["recency_weighted"]["latest"][
                "forecast_values"
            ],
        )

    def test_binary_reads_yes_probability(self) -> None:
        question = numeric_question(
            type="binary",
            aggregations={
                "recency_weighted": {
                    "latest": {"forecast_values": [0.38, 0.62]}
                }
            },
        )

        self.assertEqual(build_community_payload(question)["probability_yes"], 0.62)

    def test_multiple_choice_maps_values_to_options(self) -> None:
        question = numeric_question(
            type="multiple_choice",
            options=["up", "flat", "down"],
            aggregations={
                "recency_weighted": {
                    "latest": {"forecast_values": [0.5, 0.2, 0.3]}
                }
            },
        )

        payload = build_community_payload(question)

        self.assertEqual(
            payload["probability_yes_per_category"],
            {"up": 0.5, "flat": 0.2, "down": 0.3},
        )

    def test_invalid_numeric_cdf_is_not_interpolated_or_invented(self) -> None:
        question = numeric_question(
            aggregations={
                "recency_weighted": {
                    "latest": {"forecast_values": [0.0, 0.5, 1.0]}
                }
            }
        )

        self.assertIsNone(build_community_payload(question))


class TestEligibility(unittest.TestCase):
    def test_ready_only_after_reveal_and_before_spot_score(self) -> None:
        candidate, reason = candidate_for_question(
            numeric_question(),
            post_id=201,
            now=NOW,
            grace_after_reveal=timedelta(minutes=5),
            safety_before_score=timedelta(minutes=15),
            copy_lead=timedelta(hours=30),
        )

        self.assertEqual(reason, "ready")
        self.assertIsInstance(candidate, Candidate)

    def test_forecast_after_reveal_makes_the_run_idempotent(self) -> None:
        question = numeric_question(
            my_forecasts={
                "history": [],
                "latest": {
                    "start_time": "2026-09-18T17:00:00Z",
                    "forecast_values": [index / 200 for index in range(201)],
                },
            }
        )

        self.assertTrue(already_followed_after_reveal(question, REVEAL))
        candidate, reason = candidate_for_question(
            question,
            post_id=201,
            now=NOW,
            grace_after_reveal=timedelta(minutes=5),
            safety_before_score=timedelta(minutes=15),
            copy_lead=timedelta(hours=30),
        )
        self.assertIsNone(candidate)
        self.assertEqual(reason, "already_followed")

    def test_forecast_before_reveal_does_not_block_one_cp_copy(self) -> None:
        question = numeric_question(
            my_forecasts={
                "history": [],
                "latest": {
                    "start_time": "2026-09-17T17:00:00Z",
                    "forecast_values": [index / 200 for index in range(201)],
                },
            }
        )

        self.assertFalse(already_followed_after_reveal(question, REVEAL))

    def test_zero_length_copy_window_is_explicitly_rejected(self) -> None:
        same_time = "2026-09-18T16:00:00Z"
        candidate, reason = candidate_for_question(
            numeric_question(
                cp_reveal_time=same_time,
                spot_scoring_time=same_time,
                scheduled_close_time="2026-09-20T16:00:00Z",
            ),
            post_id=201,
            now=NOW,
            grace_after_reveal=timedelta(minutes=5),
            safety_before_score=timedelta(minutes=15),
            copy_lead=timedelta(hours=30),
        )

        self.assertIsNone(candidate)
        self.assertEqual(reason, "no_copy_window")


class FakeAPI:
    def __init__(self, questions=None, *, exists=True, username="human-user") -> None:
        self.questions = questions or []
        self.exists = exists
        self.username = username
        self.submitted = []

    def current_user(self):
        return {
            "username": self.username,
            "is_bot": False,
            "api_access_tier": "restricted",
        }

    def tournament_exists(self, slug):
        return self.exists

    def tournament_questions(self, slug):
        return self.questions

    def submit(self, candidate):
        self.submitted.append(candidate.question_id)


class TestRun(unittest.TestCase):
    def test_missing_tournament_is_a_clean_wait(self) -> None:
        api = FakeAPI(exists=False)

        exit_code = run(
            api=api,
            tournament="market-pulse-26q4",
            expected_username="human-user",
            submit=True,
            now=NOW,
            grace_minutes=5,
            safety_minutes=15,
            lead_hours=30,
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(api.submitted, [])

    def test_submit_posts_each_ready_question_once(self) -> None:
        api = FakeAPI(questions=[(201, numeric_question())])

        exit_code = run(
            api=api,
            tournament="market-pulse-26q4",
            expected_username="human-user",
            submit=True,
            now=NOW,
            grace_minutes=5,
            safety_minutes=15,
            lead_hours=30,
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(api.submitted, [101])

    def test_missing_cp_fails_without_forecasting(self) -> None:
        question = numeric_question(aggregations={})
        api = FakeAPI(questions=[(201, question)])

        exit_code = run(
            api=api,
            tournament="market-pulse-26q4",
            expected_username="human-user",
            submit=True,
            now=NOW,
            grace_minutes=5,
            safety_minutes=15,
            lead_hours=30,
        )

        self.assertEqual(exit_code, 2)
        self.assertEqual(api.submitted, [])

    def test_account_guard_prevents_wrong_account(self) -> None:
        api = FakeAPI(username="wrong-user")

        with self.assertRaises(RuntimeError):
            run(
                api=api,
                tournament="market-pulse-26q4",
                expected_username="human-user",
                submit=True,
                now=NOW,
                grace_minutes=5,
                safety_minutes=15,
                lead_hours=30,
            )


if __name__ == "__main__":
    unittest.main()
