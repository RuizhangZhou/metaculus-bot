import os
import unittest
from unittest.mock import patch

os.environ.setdefault("METACULUS_TOKEN", "test-token")

from main import _notify_matrix_on_submit


class TestMatrixSubmitNotification(unittest.TestCase):
    def test_skips_empty_run_when_not_notify_always(self) -> None:
        with patch.dict(os.environ, {"MATRIX_NOTIFY_ALWAYS": "false"}):
            with patch("main.matrix_send_message") as send:
                _notify_matrix_on_submit(
                    run_mode="tournament_update",
                    forecast_reports=[],
                )

        send.assert_not_called()

    def test_sends_summary_lines_even_without_forecasts(self) -> None:
        with patch.dict(os.environ, {"MATRIX_NOTIFY_ALWAYS": "false"}):
            with patch("main.matrix_send_message") as send:
                _notify_matrix_on_submit(
                    run_mode="tournament_update",
                    forecast_reports=[],
                    extra_lines=[
                        "- market-pulse-26q3 financial gating: checked=4, queued=1"
                    ],
                )

        send.assert_called_once()
        message = send.call_args.args[0]
        self.assertIn("0 submitted, 0 failed", message)
        self.assertIn("Run summary:", message)
        self.assertIn("financial gating: checked=4, queued=1", message)

    def test_notify_always_sends_empty_run(self) -> None:
        with patch.dict(os.environ, {"MATRIX_NOTIFY_ALWAYS": "true"}):
            with patch("main.matrix_send_message") as send:
                _notify_matrix_on_submit(
                    run_mode="tournament_update",
                    forecast_reports=[],
                )

        send.assert_called_once()
        self.assertIn("0 submitted, 0 failed", send.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
