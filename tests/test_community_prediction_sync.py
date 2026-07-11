import unittest
from types import SimpleNamespace

from community_prediction_sync import (
    build_community_prediction_payload,
    sync_questions_to_community_predictions,
)


class FakeClient:
    def __init__(self) -> None:
        self.binary_posts: list[tuple[int, float]] = []
        self.numeric_posts: list[tuple[int, list[float]]] = []
        self.mc_posts: list[tuple[int, dict[str, float]]] = []

    def post_binary_question_prediction(
        self, question_id: int, prediction_in_decimal: float
    ) -> None:
        self.binary_posts.append((question_id, prediction_in_decimal))

    def post_numeric_question_prediction(
        self, question_id: int, cdf_values: list[float]
    ) -> None:
        self.numeric_posts.append((question_id, cdf_values))

    def post_multiple_choice_question_prediction(
        self, question_id: int, options_with_probabilities: dict[str, float]
    ) -> None:
        self.mc_posts.append((question_id, options_with_probabilities))


class TestCommunityPredictionSync(unittest.TestCase):
    def test_binary_community_prediction_builds_payload(self) -> None:
        question = SimpleNamespace(
            id_of_question=123,
            question_type="binary",
            community_prediction_at_access_time=0.62,
            api_json={},
        )

        payload = build_community_prediction_payload(question)

        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertEqual(payload.kind, "binary")
        self.assertEqual(payload.value, 0.62)

    def test_numeric_community_prediction_uses_latest_cdf(self) -> None:
        question = SimpleNamespace(
            id_of_question=456,
            question_type="numeric",
            api_json={
                "question": {
                    "aggregations": {
                        "recency_weighted": {
                            "latest": {"forecast_values": [0.0, 0.25, 0.9, 1.0]}
                        }
                    }
                }
            },
        )

        payload = build_community_prediction_payload(question)

        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertEqual(payload.kind, "numeric")
        self.assertEqual(payload.value, [0.0, 0.25, 0.9, 1.0])

    def test_sync_skips_pipeline_for_available_community_prediction(self) -> None:
        client = FakeClient()
        with_cp = SimpleNamespace(
            id_of_question=1,
            question_type="binary",
            community_prediction_at_access_time=0.71,
            page_url="https://example.com/with",
            api_json={},
        )
        without_cp = SimpleNamespace(
            id_of_question=2,
            question_type="binary",
            community_prediction_at_access_time=None,
            page_url="https://example.com/without",
            api_json={},
        )

        remaining, counts = sync_questions_to_community_predictions(
            client=client,
            questions=[with_cp, without_cp],
            publish=True,
        )

        self.assertEqual(client.binary_posts, [(1, 0.71)])
        self.assertEqual(remaining, [without_cp])
        self.assertEqual(counts["synced"], 1)
        self.assertEqual(counts["missing_community_prediction"], 1)

    def test_dry_run_still_skips_pipeline(self) -> None:
        client = FakeClient()
        question = SimpleNamespace(
            id_of_question=1,
            question_type="binary",
            community_prediction_at_access_time=0.42,
            page_url="https://example.com/q",
            api_json={},
        )

        remaining, counts = sync_questions_to_community_predictions(
            client=client,
            questions=[question],
            publish=False,
        )

        self.assertEqual(client.binary_posts, [])
        self.assertEqual(remaining, [])
        self.assertEqual(counts["dry_run_synced"], 1)


if __name__ == "__main__":
    unittest.main()
