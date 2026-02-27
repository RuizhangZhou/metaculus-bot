from __future__ import annotations

import argparse
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import dotenv
import requests

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from metaculus_comment_fetcher import fetch_my_comments  # noqa: E402


_REDACTED = "<redacted>"


def _metaculus_headers(token: str) -> dict[str, str]:
    token = (token or "").strip()
    if not token:
        raise RuntimeError("METACULUS_TOKEN is missing.")
    return {
        "Authorization": f"Token {token}",
        "Accept-Language": "en",
        "User-Agent": "metaculus-bot redact comments (local)",
    }


def _fetch_my_user_id(*, token: str, timeout_seconds: int = 30) -> int:
    url = "https://www.metaculus.com/api/users/me"
    headers = _metaculus_headers(token)

    last_exc: Exception | None = None
    for attempt in range(6):
        if attempt > 0:
            _sleep_with_jitter(min(30.0, 1.0 * (2 ** (attempt - 1))))

        try:
            resp = requests.get(url, headers=headers, timeout=timeout_seconds)
        except Exception as e:
            last_exc = e
            continue

        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            try:
                retry_seconds = float(retry_after) if retry_after else None
            except Exception:
                retry_seconds = None
            _sleep_with_jitter(retry_seconds if retry_seconds is not None else 5.0 * (2**attempt))
            last_exc = requests.exceptions.HTTPError(
                f"429 Too Many Requests for {url}", response=resp
            )
            continue

        if 500 <= resp.status_code < 600:
            last_exc = requests.exceptions.HTTPError(
                f"{resp.status_code} Server error for {url}", response=resp
            )
            continue

        resp.raise_for_status()
        payload = resp.json()
        if not isinstance(payload, dict):
            raise RuntimeError(f"Unexpected /users/me payload type: {type(payload)}")
        user_id = payload.get("id")
        if not isinstance(user_id, int):
            raise RuntimeError(f"Unexpected /users/me id type: {type(user_id)}")
        return user_id

    raise RuntimeError("Failed to fetch /api/users/me") from last_exc


def _dedupe_by_id(comments: list[dict]) -> list[dict]:
    deduped: dict[int, dict] = {}
    for item in comments:
        if not isinstance(item, dict):
            continue
        cid = item.get("id")
        if isinstance(cid, int) and cid not in deduped:
            deduped[cid] = item
    return list(deduped.values())


def _redact_llms_line(text: str) -> tuple[str, bool]:
    """
    Replace the entire LLMs metadata payload (which previously included API keys)
    with a short redaction marker.
    """
    llms_line_re = re.compile(
        r"(?m)^(?P<prefix>\s*\*LLMs\*:\s*)`(?P<payload>[^`\n]*)`\s*$"
    )
    new_text, n = llms_line_re.subn(r"\g<prefix>`<redacted>`", text)
    return new_text, n > 0


def _redact_key_value_secrets(text: str) -> tuple[str, bool]:
    """
    Best-effort redaction for common secret-bearing key/value patterns in text.
    """
    # Match keys that are likely secrets (including env-var style names).
    key_name = r"[A-Za-z0-9_-]*?(?:api[_-]?key|apikey|access[_-]?token|refresh[_-]?token|client[_-]?secret|secret|password|authorization|proxy[_-]?authorization)[A-Za-z0-9_-]*?"

    # Quoted values: api_key: "...." / api_key='....'
    quoted = re.compile(
        rf'(?i)(?P<k>[\'"]?{key_name}[\'"]?\s*[:=]\s*)(?P<q>[\'"])(?P<v>[^\'"\r\n]*)(?P=q)'
    )
    new_text, n1 = quoted.subn(lambda m: f"{m.group('k')}{m.group('q')}{_REDACTED}{m.group('q')}", text)

    # Unquoted values: api_key: abcdef
    unquoted = re.compile(rf"(?i)(?P<k>['\"]?{key_name}['\"]?\s*[:=]\s*)(?P<v>[^\s,}}\]]+)")
    new_text2, n2 = unquoted.subn(lambda m: f"{m.group('k')}{_REDACTED}", new_text)

    # Authorization bearer tokens embedded in free text.
    bearer = re.compile(r"(?i)(\bbearer\s+)([A-Za-z0-9._:-]{8,})")
    new_text3, n3 = bearer.subn(r"\1<redacted>", new_text2)

    token = re.compile(r"(?i)(\btoken\s+)([A-Za-z0-9._:-]{8,})")
    new_text4, n4 = token.subn(r"\1<redacted>", new_text3)

    return new_text4, (n1 + n2 + n3 + n4) > 0


def sanitize_comment_text(
    text: str,
    *,
    redact_llms_line: bool = True,
) -> tuple[str, bool]:
    new_text = text
    changed = False
    if redact_llms_line:
        new_text, did = _redact_llms_line(new_text)
        changed = changed or did
    new_text, did = _redact_key_value_secrets(new_text)
    changed = changed or did
    return new_text, changed


def contains_secret_markers(text: str) -> bool:
    """
    Cheap filter to avoid editing comments that don't contain any obvious secrets.
    """
    key_name = r"[A-Za-z0-9_-]*?(?:api[_-]?key|apikey|access[_-]?token|refresh[_-]?token|client[_-]?secret|secret|password|authorization|proxy[_-]?authorization)[A-Za-z0-9_-]*?"
    marker_re = re.compile(
        rf"(?i)(?:['\"]?{key_name}['\"]?\s*[:=])|(?:\bbearer\s+[A-Za-z0-9._:-]{{8,}})|(?:\btoken\s+[A-Za-z0-9._:-]{{8,}})"
    )
    return bool(marker_re.search(text))


def _sleep_with_jitter(base_seconds: float) -> None:
    if base_seconds <= 0:
        return
    jitter = random.uniform(0.0, min(0.25, base_seconds * 0.25))
    time.sleep(base_seconds + jitter)


def edit_comment_text(
    *,
    token: str,
    comment_id: int,
    new_text: str,
    timeout_seconds: int = 30,
    max_retries: int = 5,
    min_delay_seconds: float = 0.25,
    backoff_base_seconds: float = 1.0,
) -> None:
    url = f"https://www.metaculus.com/api/comments/{comment_id}/edit/"
    headers = _metaculus_headers(token)

    last_exc: Exception | None = None
    for attempt in range(max(0, int(max_retries)) + 1):
        if attempt > 0:
            delay = max(float(min_delay_seconds), float(backoff_base_seconds) * (2 ** (attempt - 1)))
            _sleep_with_jitter(delay)

        try:
            resp = requests.post(
                url,
                headers=headers,
                json={"text": new_text},
                timeout=int(timeout_seconds),
            )
        except Exception as e:
            last_exc = e
            continue

        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            try:
                retry_seconds = float(retry_after) if retry_after else None
            except Exception:
                retry_seconds = None
            sleep_for = retry_seconds if retry_seconds is not None else float(backoff_base_seconds) * (2**attempt)
            sleep_for = max(float(min_delay_seconds), sleep_for)
            _sleep_with_jitter(sleep_for)
            last_exc = requests.exceptions.HTTPError(f"429 Too Many Requests for {url}", response=resp)
            continue

        if 500 <= resp.status_code < 600:
            last_exc = requests.exceptions.HTTPError(
                f"{resp.status_code} Server error for {url}",
                response=resp,
            )
            continue

        try:
            resp.raise_for_status()
        except Exception as e:
            last_exc = e
            continue

        return

    raise RuntimeError(f"Failed to edit comment {comment_id}") from last_exc


def main(argv: list[str]) -> int:
    dotenv.load_dotenv()

    parser = argparse.ArgumentParser(
        description="Redact secrets from your Metaculus comments by editing them in-place."
    )
    parser.add_argument(
        "--scope",
        choices=["private", "public", "all"],
        default="all",
        help="Which comments to process (default: all).",
    )
    parser.add_argument(
        "--author-id",
        type=int,
        default=0,
        help="Optional Metaculus user id (avoids /api/users/me).",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=5000,
        help="Maximum comments to fetch per scope (default: 5000).",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.35,
        help="Base sleep between edits to avoid rate limits (default: 0.35).",
    )
    parser.add_argument(
        "--no-redact-llms-line",
        action="store_true",
        help="Do not replace the '*LLMs*:' metadata line payload.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without editing.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit of edits to apply (0 = no limit).",
    )
    args = parser.parse_args(argv)

    token = os.getenv("METACULUS_TOKEN", "").strip()
    if not token:
        print("Error: METACULUS_TOKEN is missing.", file=sys.stderr)
        return 2

    author_id_env = os.getenv("METACULUS_AUTHOR_ID", "").strip()
    author_id = int(args.author_id) if int(args.author_id) > 0 else 0
    if author_id <= 0 and author_id_env:
        try:
            author_id = int(author_id_env)
        except Exception:
            author_id = 0

    if author_id > 0:
        my_user_id = author_id
    else:
        try:
            my_user_id = _fetch_my_user_id(token=token)
        except Exception as e:
            print(f"Error: failed to fetch /api/users/me: {e}", file=sys.stderr)
            print(
                "Tip: pass --author-id <id> or set METACULUS_AUTHOR_ID to avoid this call.",
                file=sys.stderr,
            )
            return 2

    max_items = max(1, int(args.max_items))
    all_comments: list[dict[str, Any]] = []
    try:
        if args.scope in {"private", "all"}:
            all_comments.extend(
                fetch_my_comments(
                    token=token,
                    author_id=my_user_id,
                    max_items=max_items,
                    include_private=True,
                )
            )
        if args.scope in {"public", "all"}:
            all_comments.extend(
                fetch_my_comments(
                    token=token,
                    author_id=my_user_id,
                    max_items=max_items,
                    include_private=False,
                )
            )
    except Exception as e:
        print(f"Error: failed to fetch comments: {e}", file=sys.stderr)
        return 2

    comments = _dedupe_by_id(all_comments)
    # Newest first for faster confidence.
    comments.sort(key=lambda c: (c.get("created_at") or ""), reverse=True)

    to_edit: list[tuple[int, str]] = []
    redact_llms_line = not bool(args.no_redact_llms_line)
    for comment in comments:
        if not isinstance(comment, dict):
            continue
        if comment.get("is_soft_deleted") is True:
            continue
        cid = comment.get("id")
        text = comment.get("text")
        if not isinstance(cid, int) or not isinstance(text, str):
            continue

        if not contains_secret_markers(text):
            continue

        new_text, changed = sanitize_comment_text(text, redact_llms_line=redact_llms_line)
        if changed and new_text != text:
            to_edit.append((cid, new_text))

    limit = int(args.limit)
    if limit > 0:
        to_edit = to_edit[:limit]

    print(f"Fetched {len(comments)} comments ({args.scope}). Will edit {len(to_edit)}.")
    if args.dry_run:
        for cid, _ in to_edit[:50]:
            print(f"- would_edit comment_id={cid}")
        if len(to_edit) > 50:
            print(f"... and {len(to_edit) - 50} more")
        return 1 if to_edit else 0

    edited = 0
    failed: list[int] = []
    for i, (cid, new_text) in enumerate(to_edit, start=1):
        try:
            edit_comment_text(token=token, comment_id=cid, new_text=new_text)
            edited += 1
            print(f"[{i}/{len(to_edit)}] edited comment_id={cid}")
        except Exception as e:
            failed.append(cid)
            print(f"[{i}/{len(to_edit)}] FAILED comment_id={cid}: {type(e).__name__}: {e}", file=sys.stderr)
        _sleep_with_jitter(float(args.sleep_seconds))

    print(f"Done. Edited {edited}/{len(to_edit)}. Failed: {len(failed)}.")
    if failed:
        print("Failed IDs:", ",".join(str(x) for x in failed), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
