# Issue: Local question-link crawl (Playwright) + keep Exa

## Context
The bot currently relies on web search providers (e.g. Exa via `SmartSearcher`) to gather external information.
However, Metaculus questions often already include highly relevant links in the background/resolution criteria/fine print.

We want a **local, production-oriented retrieval step** that:
- renders pages (incl. JS-heavy) locally via headless Chromium
- extracts main content (Reader/Readability-style)
- returns sanitized plain text with hard budgets
- can be used **before** (and alongside) Exa search

## Goal
Add an optional "local crawl" step that fetches:
1) the Metaculus question page (optional), and
2) all explicit `http(s)://...` links in the question text fields,
then provides extracted clean text to the research prompt.

Exa remains available and is still used for broader web search.

## In scope
- Playwright-based fetch/render (Chromium headless)
- Main-content extraction (Readability-like, with fallback)
- Text sanitization + normalization
- Hard limits: timeouts + total/per-source char budgets + truncation marker
- Concurrency limits + browser reuse within an async run
- Logging consistent with current bot logging
- Unit tests for URL extraction + normalization + truncation
- Optional integration test gated by env/deps

## Out of scope (for now)
- Full caching layer across runs
- CAPTCHA / bot-protection solving
- Domain-specific extraction rules
- PDF parsing

## Acceptance criteria
- No external HTML-to-text proxy service is used.
- Drop-in: existing forecast flow still works when local crawl is disabled.
- When enabled, local crawl runs without leaking secrets, and failures do not crash the bot (unless explicitly configured).
- JS-heavy pages are supported (rendered with Playwright).
- Output is plain text (no HTML/script).
- Budgets are enforced:
  - configurable timeouts
  - default 20,000 total chars (with truncation marker)
- Under load: browser is reused per async run and page fetch concurrency is bounded.

