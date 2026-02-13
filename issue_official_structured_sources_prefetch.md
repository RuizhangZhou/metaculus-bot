# Issue: Add official structured sources prefetch (FRED/BLS/BEA/EIA/Federal Register/NOAA/USGS) routed by ToolRouter

## Context
We already have a research flow that prioritizes:
1) **Local question-link crawl** (Metaculus question page + explicit links, rendered locally via Playwright when enabled)
2) **Free official deterministic sources** (currently: SEC EDGAR companyfacts + filings submissions; Nasdaq EPS endpoints)
3) **Web search (Exa/SmartSearcher or browsing models)** as a last resort

The goal of this issue is to expand step (2) with additional **high-signal, official, structured, stable** public sources,
and make them available as tools that the **LLM ToolRouter can choose per question** (no manual toggles per run).

## Goal
Add a set of official-source prefetchers (each with hard budgets + robust error handling) and integrate them with:
- **ToolRouter** (decides whether to fetch each source)
- existing research prompt assembly (official blocks before any web search)

## Sources (initial set)
### Macro / econ / labor
- **FRED** (Fed St. Louis) time series (rates, inflation, employment, GDP proxies)
- **BLS** (CPI, unemployment, employment, etc.)
- **BEA** (GDP, PCE, trade, etc.)

### Energy / commodities
- **EIA** (oil inventories, production, prices, etc.)

### Regulation / policy (verifiable timestamps)
- **Federal Register** API (rules, notices, effective dates, docket references)

### Natural hazards / environment
- **NOAA / NHC** (hurricane advisories / tracks; prefer RSS/official endpoints)
- **USGS** (earthquakes feeds)

## API keys / configuration
Some sources typically require a free API key or benefit from one for higher limits:
- `FRED_API_KEY` (often required)
- `BLS_API_KEY` (often optional but improves quotas)
- `BEA_API_KEY`
- `EIA_API_KEY`
- `NOAA_API_TOKEN` (depends on endpoint; NHC RSS usually no key)

Non-key sources (initially):
- Federal Register API (no key)
- USGS earthquake feeds (no key)

Design constraint: **router decides when to call tools**, but endpoints must still respect **provider policy** and budget limits.

## Proposed architecture
### 1) Official source fetchers
Implement one module per source (or a single `official_sources.py`) with functions like:
- `prefetch_fred(question) -> str`
- `prefetch_bls(question) -> str`
- `prefetch_bea(question) -> str`
- `prefetch_eia(question) -> str`
- `prefetch_federal_register(question) -> str`
- `prefetch_noaa_nhc(question) -> str`
- `prefetch_usgs_earthquakes(question) -> str`

Each fetcher must:
- be **deterministic** (no LLM summarization inside)
- return **concise** structured extracts + **source URLs**
- enforce **timeouts**, **max items**, **max chars**, and **rate-friendly** behavior
- handle **Cancellation** cleanly
- fail soft (empty string) unless explicitly configured otherwise

### 2) ToolRouter schema extension
Extend `ToolRouterPlan` to include booleans for new sources, e.g.:
- `fetch_fred`
- `fetch_bls`
- `fetch_bea`
- `fetch_eia`
- `fetch_federal_register`
- `fetch_noaa_nhc`
- `fetch_usgs`

Router prompt should emphasize:
- prefer local extracts + official sources first
- only use web search if needed
- prefer official sources when resolution criteria reference them (or when topic matches)

### 3) Caching & budgets
- In-run caching via Notepad per question (like local crawl + router plan)
- Optional TTL caching (like SEC submissions cache) for endpoints that are expensive or rate-limited

## Acceptance criteria
- Fetchers are integrated into `run_research` official block generation (before any web search).
- ToolRouter can selectively enable/disable each source per question.
- Missing keys do not crash the bot; fetchers return empty + log a concise warning.
- Hard budgets: timeouts, max chars per source, max total official chars.
- Tests cover:
  - budget/truncation behavior
  - missing-key behavior (no network calls)
  - basic parsing/formatting for at least one “no-key” source (Federal Register / USGS) using mocked HTTP responses

## Out of scope (initially)
- Full caching across runs / persistent DB cache
- Domain-specific allowlists/heuristics beyond simple topic matching + router decision
- PDF parsing of filings or complex HTML-to-text beyond current local crawl module
- CAPTCHA / bot-protection bypass

