# Planned GitHub Issues

Run `gh auth login` first, then create these with the commands below.

---

## 1. Backtest Harness for Parameter Tuning

```bash
gh issue create --title "Build backtest harness for parameter tuning" --body "$(cat <<'EOF'
## Problem
Currently there is no systematic way to compare different bot configurations (model choice, search depth, calibration alpha, research count). Parameter decisions are made by intuition, not data.

## Proposed Solution
Build a backtest harness that:
1. Maintains a fixed set of already-resolved Metaculus questions (diverse types: binary, numeric, date, MC)
2. Runs the bot with different configs against these questions (without submitting)
3. Records per-config metrics: Brier score, log score, relative-to-CP improvement, cost, latency
4. Outputs a comparison table and optionally a Pareto frontier (accuracy vs cost)

## Key Design Points
- Store resolved questions as JSON fixtures (question text + resolution + CP at access time)
- Support A/B comparison: `python backtest.py --config-a default --config-b deep-research`
- Track token usage per run for cost estimation
- Separate "research quality" score (did the research find the key fact?) from "calibration quality" (was the probability well-placed?)

## Acceptance Criteria
- [ ] 50+ resolved questions across binary/numeric/date/MC types
- [ ] CLI that runs bot against fixtures and outputs score table
- [ ] At least one A/B comparison documented in repo
EOF
)"
```

## 2. Multi-Model Ensemble with Diverse Perspectives

```bash
gh issue create --title "Implement true multi-model ensemble with diverse perspectives" --body "$(cat <<'EOF'
## Problem
Current bot runs the same prompt N times with the same model. This reduces variance but cannot fix systematic bias. Real forecasting ensembles need diversity of perspective, not just sampling noise.

## Proposed Solution
Implement 3 distinct forecasting perspectives, each with its own prompt and optionally different model:
1. **Outside-view agent**: Base rates, reference classes, historical analogies. No current news.
2. **Inside-view agent**: Current news, recent developments, expert opinions. Uses web search heavily.
3. **Market/data agent**: Prediction market prices, official data APIs, structured sources.

Aggregate using:
- Geometric mean of odds (log-linear pool) for binary
- Quantile averaging for numeric/date
- Optional extremization factor calibrated from backtest data

## Key Design Points
- Each agent gets a different system prompt emphasizing its perspective
- Can use different models (e.g., Claude for outside-view, GPT for inside-view)
- Aggregation weights can be learned from backtest harness (#1)
- Fall back to single-agent mode when budget is tight

## Acceptance Criteria
- [ ] 3 distinct forecaster prompts with different perspectives
- [ ] Aggregation function (log-linear pool + extremization)
- [ ] Config to enable/disable per-agent and set weights
- [ ] Backtest comparison: ensemble vs single-agent
EOF
)"
```

## 3. Historical Calibration Learning

```bash
gh issue create --title "Replace hardcoded calibration with learned logit blend" --body "$(cat <<'EOF'
## Problem
Current calibration logic (`_calibrate_binary_forecast`) uses hardcoded trust=[0.10, 0.60] to blend bot prediction toward community prediction, defaulting CP=0.5 when missing. This:
- Drags predictions toward 0.5 for new questions (no CP yet)
- Cannot learn from the bot's actual track record
- Uses the same blend regardless of question type

## Proposed Solution
1. **Short-term fix** (this sprint): When CP is missing, skip the CP blend entirely and only apply the extremeness clamp. Don't invent a fake anchor.
2. **Medium-term**: Fit a logistic regression on resolved questions: `calibrated_logit = a * raw_logit + b` where a,b are learned from historical (raw_prediction, actual_resolution) pairs.
3. **Long-term**: Per-category calibration (politics, science, economics, etc.) with enough data.

## Key Design Points
- Store raw predictions and resolutions in a calibration dataset (JSON/CSV)
- Retrain calibration curve weekly as part of weekly retrospective
- Support separate curves for binary, MC, numeric
- Validate with cross-validation on historical data

## Acceptance Criteria
- [ ] Short-term: CP-missing case fixed (no drag toward 0.5)
- [ ] Calibration dataset auto-populated from retrospective runs
- [ ] Fitted calibration curve applied at prediction time
- [ ] Cross-validated Brier score improvement documented
EOF
)"
```

## 4. Multi-Perspective Structured Research Pipeline

```bash
gh issue create --title "Structured multi-query research pipeline" --body "$(cat <<'EOF'
## Problem
Current research uses a single broad web search query and asks the LLM to cover everything in one shot. This misses:
- Base rates and reference classes
- Contradictory evidence
- Official primary sources vs news summaries
- Prediction market prices (often missed or hallucinated)

## Proposed Solution
Replace single-query research with a structured pipeline:
1. **Resolution criteria analysis**: Parse what exactly needs to happen for each resolution
2. **Base rate / outside view**: "How often does [event class] happen?" (no web search needed)
3. **Current status query**: Targeted web search for latest developments
4. **Official source query**: Check named sources from resolution criteria
5. **Prediction market query**: Explicit search for Polymarket/Kalshi/Manifold prices
6. **Contradiction check**: Search for counter-evidence to the leading hypothesis

Each step generates 1-2 targeted queries instead of 1 broad query.

## Key Design Points
- Steps 1-2 can use cheap/free models (no web search)
- Steps 3-5 each get their own web search call (3-5 total vs current 1)
- Step 6 is optional, triggered when confidence is high (>85% or <15%)
- Total cost increase ~3x search calls but much higher information gain
- Compatible with tool router (router decides which steps to skip)

## Acceptance Criteria
- [ ] Research pipeline with distinct query stages
- [ ] Each stage produces labeled output section in research report
- [ ] Backtest shows improved research quality (key facts found more often)
- [ ] Cost stays within 5x of current per-question budget
EOF
)"
```
