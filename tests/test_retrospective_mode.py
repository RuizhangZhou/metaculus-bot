import unittest
from datetime import timezone

from retrospective_mode import (
    _parse_iso8601_utc,
    _resolution_to_binary_outcome,
    _score_binary,
    _select_api2_question_json,
)


class TestParseIso8601Utc(unittest.TestCase):
    def test_parses_z_suffix(self) -> None:
        dt = _parse_iso8601_utc("2026-02-05T15:04:42.419199Z")
        assert dt is not None
        self.assertEqual(dt.tzinfo, timezone.utc)
        self.assertEqual(dt.year, 2026)


class TestBinaryResolution(unittest.TestCase):
    def test_resolution_to_binary_outcome(self) -> None:
        self.assertEqual(_resolution_to_binary_outcome("yes"), 1.0)
        self.assertEqual(_resolution_to_binary_outcome("no"), 0.0)
        self.assertEqual(_resolution_to_binary_outcome(True), 1.0)
        self.assertEqual(_resolution_to_binary_outcome(False), 0.0)
        self.assertEqual(_resolution_to_binary_outcome(1), 1.0)
        self.assertEqual(_resolution_to_binary_outcome(0), 0.0)
        self.assertIsNone(_resolution_to_binary_outcome("unknown"))

    def test_score_binary(self) -> None:
        self.assertEqual(_score_binary(prediction=0.25, outcome=1.0), (0.25 - 1.0) ** 2)
        self.assertEqual(_score_binary(prediction=0.25, outcome=0.0), (0.25 - 0.0) ** 2)
        self.assertIsNone(_score_binary(prediction="0.2", outcome=1.0))
        self.assertIsNone(_score_binary(prediction=2.0, outcome=1.0))
        self.assertIsNone(_score_binary(prediction=0.5, outcome=None))


class TestSelectApi2QuestionJson(unittest.TestCase):
    def test_selects_normal_question(self) -> None:
        payload = {"question": {"id": 123, "type": "binary"}}
        out = _select_api2_question_json(api2_post=payload, question_id=123)
        self.assertEqual(out, payload["question"])

    def test_selects_group_subquestion(self) -> None:
        payload = {
            "group_of_questions": {
                "questions": [
                    {"id": 1, "type": "binary"},
                    {"id": 2, "type": "binary"},
                ]
            }
        }
        out = _select_api2_question_json(api2_post=payload, question_id=2)
        assert out is not None
        self.assertEqual(out.get("id"), 2)
