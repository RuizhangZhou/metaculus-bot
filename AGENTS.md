# Codex Agent Instructions (metaculus-bot)

## PR / Merge policy (IMPORTANT)
- Default behavior: create a branch + open a PR, **do not merge**.
- Only merge to `main` when the user explicitly asks in the current conversation turn (e.g. “merge/合并到 main/帮我 merge”).
- If the user asks for a PR but does not explicitly ask to merge, leave the PR open and request confirmation before merging.
- Do not push directly to `main` unless explicitly requested.

## Secrets
- Never print or paste any tokens/keys/secrets in responses or logs.
