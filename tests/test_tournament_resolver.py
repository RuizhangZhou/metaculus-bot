import unittest
from datetime import datetime, timezone

import tournament_resolver as tr


NOW = datetime(2026, 8, 28, 12, 0, tzinfo=timezone.utc)

# Shapes taken from the real /api/projects/tournaments/ listing.
LISTING = [
    {
        "slug": "market-pulse-26q2",
        "type": "tournament",
        "bot_leaderboard_status": "include",
        "start_date": "2026-03-30T00:00:00Z",
        "close_date": "2026-07-01T00:00:00Z",
    },
    {
        "slug": "market-pulse-26q3",
        "type": "tournament",
        "bot_leaderboard_status": "include",
        "start_date": "2026-07-08T21:00:00Z",
        "close_date": "2026-10-01T00:00:00Z",
    },
    {
        "slug": "summer-futureeval-2026",
        "type": "tournament",
        "bot_leaderboard_status": "bots_only",
        "start_date": "2026-05-18T00:00:00Z",
        "forecasting_end_date": "2026-09-06T00:00:00Z",
        "close_date": "2026-11-05T00:00:00Z",
        "is_ongoing": True,
    },
    {
        "slug": "metaculus-cup-summer-2026",
        "type": "tournament",
        "bot_leaderboard_status": "exclude_and_show",
        "start_date": "2026-05-04T00:00:00Z",
        "close_date": "2026-09-04T00:00:00Z",
    },
    {
        "slug": None,  # the listing really does contain these
        "type": "question_series",
        "bot_leaderboard_status": "exclude_and_show",
        "start_date": None,
        "close_date": None,
    },
]


class TestResolveFamily(unittest.TestCase):
    def test_market_pulse_picks_only_the_running_quarter(self) -> None:
        self.assertEqual(
            tr.resolve_family("market-pulse", now=NOW, listing=LISTING),
            ["market-pulse-26q3"],
        )

    def test_seasonal_bot_tournament_matched_on_bots_only_not_on_name(self) -> None:
        self.assertEqual(
            tr.resolve_family("futureeval", now=NOW, listing=LISTING),
            ["summer-futureeval-2026"],
        )

    def test_next_season_is_picked_up_without_any_config_change(self) -> None:
        listing = LISTING + [
            {
                "slug": "fall-futureeval-2026",
                "type": "tournament",
                "bot_leaderboard_status": "bots_only",
                "start_date": "2026-09-28T00:00:00Z",
                "forecasting_end_date": "2027-01-06T00:00:00Z",
                "close_date": "2027-03-05T00:00:00Z",
                "is_ongoing": True,
            }
        ]
        # This is before Fall's advertised start, but its practice question is
        # already open and the API marks the tournament ongoing.  Summer's
        # resolution period continues, but its forecasting window is over.
        prestart = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)
        self.assertEqual(
            tr.resolve_family("futureeval", now=prestart, listing=listing),
            ["fall-futureeval-2026"],
        )

    def test_future_prestart_tournament_is_not_selected_until_ongoing(self) -> None:
        listing = LISTING + [
            {
                "slug": "winter-futureeval-2027",
                "type": "tournament",
                "bot_leaderboard_status": "bots_only",
                "start_date": "2027-01-20T00:00:00Z",
                "forecasting_end_date": "2027-04-20T00:00:00Z",
                "close_date": "2027-06-01T00:00:00Z",
                "is_ongoing": False,
            }
        ]
        self.assertEqual(
            tr.resolve_family("futureeval", now=NOW, listing=listing),
            ["summer-futureeval-2026"],
        )

    def test_overlapping_quarters_both_returned_oldest_first(self) -> None:
        listing = LISTING + [
            {
                "slug": "market-pulse-26q4",
                "type": "tournament",
                "bot_leaderboard_status": "include",
                "start_date": "2026-09-28T00:00:00Z",
                "close_date": "2027-01-07T00:00:00Z",
            }
        ]
        overlap = datetime(2026, 9, 29, 12, 0, tzinfo=timezone.utc)
        self.assertEqual(
            tr.resolve_family("market-pulse", now=overlap, listing=listing),
            ["market-pulse-26q3", "market-pulse-26q4"],
        )

    def test_gap_between_quarters_returns_nothing_rather_than_guessing(self) -> None:
        gap = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)  # q3 closed, q4 not up
        self.assertEqual(tr.resolve_family("market-pulse", now=gap, listing=LISTING), [])

    def test_minibench_slug_is_constant_and_needs_no_listing(self) -> None:
        self.assertEqual(tr.resolve_family("minibench", now=NOW, listing=[]), ["minibench"])

    def test_unknown_family_is_an_error_not_a_silent_empty(self) -> None:
        with self.assertRaises(ValueError):
            tr.resolve_family("no-such-family", now=NOW, listing=LISTING)


class TestExpandIdentifiers(unittest.TestCase):
    def test_auto_uses_the_default_family(self) -> None:
        self.assertEqual(
            tr.expand_identifiers(
                ["auto"], default_family="market-pulse", now=NOW, listing=LISTING
            ),
            ["market-pulse-26q3"],
        )

    def test_auto_with_explicit_family_overrides_the_default(self) -> None:
        self.assertEqual(
            tr.expand_identifiers(
                ["auto:futureeval"], default_family="market-pulse", now=NOW, listing=LISTING
            ),
            ["summer-futureeval-2026"],
        )

    def test_explicit_slugs_pass_through_untouched(self) -> None:
        self.assertEqual(
            tr.expand_identifiers(
                ["market-pulse-26q1", "auto:minibench"],
                default_family="market-pulse",
                now=NOW,
                listing=LISTING,
            ),
            ["market-pulse-26q1", "minibench"],
        )

    def test_duplicates_are_dropped_case_insensitively(self) -> None:
        self.assertEqual(
            tr.expand_identifiers(
                ["auto", "Market-Pulse-26Q3"],
                default_family="market-pulse",
                now=NOW,
                listing=LISTING,
            ),
            ["market-pulse-26q3"],
        )

    def test_mixed_auto_markers_resolve_together(self) -> None:
        self.assertEqual(
            tr.expand_identifiers(
                ["auto:futureeval", "auto:minibench"],
                default_family="futureeval",
                now=NOW,
                listing=LISTING,
            ),
            ["summer-futureeval-2026", "minibench"],
        )

    def test_is_auto_recognises_the_markers(self) -> None:
        self.assertTrue(tr.is_auto("auto"))
        self.assertTrue(tr.is_auto(" AUTO:market-pulse "))
        self.assertFalse(tr.is_auto("market-pulse-26q3"))
        self.assertFalse(tr.is_auto(""))


if __name__ == "__main__":
    unittest.main()
