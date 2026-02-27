# Codex Agent Instructions (metaculus-bot)

## PR / Merge policy (IMPORTANT)
- Default behavior: create a branch + open a PR, **do not merge**.
- Only merge to the repository's default branch (currently `main`) when the user explicitly asks to merge in their message in this conversation (e.g. `merge`, `合并到 main`, `帮我 merge`).
- If the user asks for a PR but does not explicitly ask to merge, leave the PR open and request confirmation before merging.
- Do not push directly to the default branch (currently `main`) unless explicitly requested.

## Secrets
- Never print or paste any tokens/keys/secrets in responses or logs.

## Commit message convention
- Include explicit identifiers for both the issue and PR to avoid ambiguity (GitHub uses `#<number>` for both).
- Format: `<type>: <summary> (issue #<issue>, PR #<pr>)`
- Example: `fix: tighten action timeouts (issue #19, PR #21)`
