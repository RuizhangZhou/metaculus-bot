# Market Pulse 26Q4 participation guide

Snapshot date: 2026-09-14 UTC. The `market-pulse-26q4` project was not yet in
the authenticated tournament listing at this point, so every Q4-specific rule
and timestamp must be checked again when the project appears.

## What 26Q3 actually did

This is calculated from all 68 Q3 subquestions returned by the authenticated
Metaculus API, using each subquestion's own `open_time`, `cp_reveal_time`,
`scheduled_close_time`, and `spot_scoring_time`.

| Question family | Count | Open pattern | CP reveal | Spot score |
|---|---:|---|---|---|
| Rolling biweekly groups | 48 (6 waves × 8) | Usually Monday 16:00 UTC, every 14 days | Friday 16:00 UTC, 96h after open | Monday 03:00 UTC, 59h after reveal |
| First rolling wave | included above | Wed 2026-07-08 21:00 UTC | Fri 2026-07-10 16:00 UTC | Mon 2026-07-13 03:00 UTC |
| Event/earnings questions | 20 | Irregular | Usually 10:00/12:00 UTC | Usually 28–48h after reveal |

Across all 68 questions, reveal-to-spot-score intervals were:

- 48 questions: 59h
- 14 questions: about 46h (13 exactly, one 46h01m)
- 3 questions: 48h
- 2 questions: 28h
- 1 annulled question: 0h

There were 14 distinct spot-scoring timestamps. `spot_scoring_time` equalled
`scheduled_close_time` on 66/68 questions, but not on these two:

- Amazon EPS, qid 44793: CP revealed 2026-07-29 12:00 UTC; spot-scored
  2026-07-30 16:00 UTC; scheduled close was 2026-07-31 10:00 UTC.
- Nvidia revenue, qid 45364 (annulled): CP reveal and spot score were both
  2026-08-24 12:00 UTC; scheduled close was 2026-08-26 10:00 UTC.

Therefore, automation must use `spot_scoring_time`, not a tournament-level
date and not blindly `scheduled_close_time`.

## Is one submission enough?

Yes. Market Pulse uses spot scoring: only the forecast standing at the spot
scoring timestamp is evaluated. A single submission per question is enough if
it is made after the CP is visible, remains active, and lands before the spot
timestamp. The CP can move after it is first revealed, so this implementation
waits until the final 30 hours rather than copying immediately.

The API schedule itself is the queue. Persistent queue state is unnecessary:
`my_forecasts.history` tells an ephemeral runner whether the account already
submitted after that question's CP reveal. This also makes retries idempotent.

## Low-cost GitHub Actions design

`run_market_pulse_cp_follower.yaml` runs once daily at 17:20 UTC. This is after
every Q3 reveal hour (10:00, 12:00, or 16:00 UTC). For each exact Q4
subquestion it:

1. waits until at least five minutes after `cp_reveal_time`;
2. waits until `spot_scoring_time` is no more than 30 hours away;
3. requires at least 15 minutes of safety before the spot timestamp;
4. skips if the account already forecast after CP reveal;
5. copies the aggregate exactly (including the raw 201-point continuous CDF);
6. fails if CP data is absent, with no LLM/research/fallback prediction.

It uses only checkout, Python, and the stdlib—no Poetry, browser, LLM, search,
or cache setup. Scheduled jobs are gated by
`MARKET_PULSE_CP_FOLLOW_ENABLED=true`; while the variable is absent/false,
GitHub does not allocate a runner for scheduled events.

## Important API blocker

As of the snapshot date, the existing `METACULUS_TOKEN` belongs to
`ruizhangzhou-bot` (`is_bot=true`, `api_access_tier=restricted`). Its API
response contained no current or historical Community Prediction on any of the
68 Q3 subquestions. Metaculus's API documentation says ordinary authenticated
accounts receive current CP data only on a small allowlisted question set.

This makes unattended copying **not reliable with the current token**. Do not
enable submission merely by reusing the FutureEval secret.

## Recommended Q4 setup

Participate manually from one normal account unless/until a dry-run proves
that the same account's API token can read Q4 CP values. Q3 rules allowed one
entry per tournament—either bot or normal user, never both—so verify the same
wording on Q4 before making the first forecast.

For guarded automation after access is confirmed:

1. Create a separate Actions secret `MARKET_PULSE_METACULUS_TOKEN` containing
   the chosen Q4 account's token.
2. Set `MARKET_PULSE_EXPECTED_USERNAME` to that exact username. The script
   refuses a different account.
3. Keep `MARKET_PULSE_CP_TOURNAMENT=market-pulse-26q4` (the exact slug avoids
   accidentally submitting to overlapping Q3).
4. After Q4 launches and a CP has been revealed, manually dispatch a dry-run:
   `gh workflow run run_market_pulse_cp_follower.yaml --ref main -f tournament=market-pulse-26q4 -f submit=false`.
5. Confirm the run reports a `ready` question rather than
   `community_prediction_unavailable`.
6. Only then set `MARKET_PULSE_CP_FOLLOW_ENABLED=true`. Scheduled events submit;
   manual dispatch stays dry-run unless `submit=true` is selected.

For manual participation, follow the tournament and use the existing server
watcher/Matrix alert. On every alert, open each linked subquestion, read its
currently visible CP, submit that value once, and verify the displayed spot
scoring time rather than relying on the outer tournament close date.

## Primary references

- [Market Pulse Challenge 26Q3](https://www.metaculus.com/tournament/market-pulse-26q3/)
- [Metaculus scoring FAQ](https://www.metaculus.com/help/scores-faq/)
- [Metaculus API specification](https://github.com/Metaculus/metaculus/blob/main/docs/openapi.yml)

