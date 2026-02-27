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


def _metaculus_headers(token: str) -> dict[str, str]:
    token = (token or "").strip()
    if not token:
        raise RuntimeError("METACULUS_TOKEN is missing.")
    return {
        "Authorization": f"Token {token}",
        "Accept-Language": "en",
        "User-Agent": "metaculus-bot comment audit (local)",
    }


def _fetch_my_user_id(*, token: str, timeout_seconds: int = 30) -> int:
    url = "https://www.metaculus.com/api/users/me"
    headers = _metaculus_headers(token)
    last_exc: Exception | None = None
    for attempt in range(6):
        if attempt > 0:
            base = min(30.0, 1.0 * (2 ** (attempt - 1)))
            jitter = random.uniform(0.0, min(0.25, base * 0.25))
            time.sleep(base + jitter)

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
            base = retry_seconds if retry_seconds is not None else 5.0 * (2**attempt)
            jitter = random.uniform(0.0, min(0.25, base * 0.25))
            time.sleep(base + jitter)
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


def _find_secret_markers(text: str) -> list[str]:
    lowered = text.lower()
    markers = [
        "api_key",
        "access_token",
        "refresh_token",
        "client_secret",
        "authorization",
        "bearer ",
        "chat.kiconnect.nrw",
    ]
    return [m for m in markers if m in lowered]


def _is_redacted_value(value: str) -> bool:
    v = (value or "").strip().strip("'\"").strip()
    if not v:
        return True
    lowered = v.lower()
    if "<redacted>" in lowered or "[redacted]" in lowered or "redacted" == lowered:
        return True
    if set(lowered) <= {"x"} and len(lowered) >= 6:
        return True
    return False


def _find_unredacted_sensitive_assignments(text: str) -> list[str]:
    """
    Returns a list of sensitive keys that appear to have NON-redacted values.
    """
    key_name = r"[A-Za-z0-9_-]*?(?:api[_-]?key|apikey|access[_-]?token|refresh[_-]?token|client[_-]?secret|secret|password|authorization|proxy[_-]?authorization)[A-Za-z0-9_-]*?"

    quoted = re.compile(
        rf'(?i)(?P<k>[\'"]?{key_name}[\'"]?\s*[:=]\s*)(?P<q>[\'"])(?P<v>[^\'"\r\n]*)(?P=q)'
    )
    unquoted = re.compile(rf"(?i)(?P<k>['\"]?{key_name}['\"]?\s*[:=]\s*)(?P<v>[^\s,}}\]]+)")
    hits: list[str] = []
    for m in quoted.finditer(text):
        key_part = m.group("k")
        val = m.group("v")
        if not _is_redacted_value(val):
            hits.append(key_part.strip())
    for m in unquoted.finditer(text):
        key_part = m.group("k")
        val = m.group("v")
        if not _is_redacted_value(val):
            hits.append(key_part.strip())

    # Bearer/Token strings in free text (e.g., "Authorization: Bearer abc...")
    bearer = re.compile(r"(?i)\bbearer\s+([A-Za-z0-9._:-]{8,})")
    for m in bearer.finditer(text):
        if not _is_redacted_value(m.group(1)):
            hits.append("bearer")
    token = re.compile(r"(?i)\btoken\s+([A-Za-z0-9._:-]{8,})")
    for m in token.finditer(text):
        if not _is_redacted_value(m.group(1)):
            hits.append("token")

    # Dedupe while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for h in hits:
        if h in seen:
            continue
        seen.add(h)
        out.append(h)
    return out


def _comment_question_url(on_post: object) -> str:
    if isinstance(on_post, int):
        return f"https://www.metaculus.com/questions/{on_post}/"
    return ""


def _dedupe_by_id(comments: list[dict]) -> list[dict]:
    deduped: dict[int, dict] = {}
    for item in comments:
        if not isinstance(item, dict):
            continue
        cid = item.get("id")
        if isinstance(cid, int) and cid not in deduped:
            deduped[cid] = item
    return list(deduped.values())


def main(argv: list[str]) -> int:
    dotenv.load_dotenv()

    parser = argparse.ArgumentParser(
        description=(
            "Audit your Metaculus comments for leaked secrets (prints IDs only; never prints comment text)."
        )
    )
    parser.add_argument(
        "--scope",
        choices=["private", "public", "all"],
        default="all",
        help="Which comments to scan (default: all).",
    )
    parser.add_argument(
        "--mode",
        choices=["leaks", "mentions"],
        default="leaks",
        help="Scan mode: 'leaks' finds non-redacted secrets; 'mentions' flags any keyword mentions.",
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
        default=1000,
        help="Maximum comments to fetch per scope (default: 1000).",
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

    all_comments: list[dict[str, Any]] = []
    max_items = max(1, int(args.max_items))

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
    findings: list[tuple[dict, list[str]]] = []
    for comment in comments:
        text = comment.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        if args.mode == "mentions":
            markers = _find_secret_markers(text)
        else:
            markers = _find_unredacted_sensitive_assignments(text)
        if markers:
            findings.append((comment, markers))

    findings.sort(key=lambda item: (item[0].get("created_at") or ""), reverse=True)

    total = len(comments)
    label = "Potential secret-leak matches" if args.mode == "mentions" else "Unredacted secret matches"
    print(f"Scanned {total} comments ({args.scope}). {label}: {len(findings)}")
    for comment, markers in findings:
        cid = comment.get("id")
        on_post = comment.get("on_post")
        is_private = comment.get("is_private")
        created_at = comment.get("created_at")
        url = _comment_question_url(on_post)
        markers_str = ",".join(markers)
        print(
            f"- comment_id={cid} post_id={on_post} is_private={is_private} created_at={created_at} markers={markers_str} {url}"
        )

    # Non-zero exit code if anything matched (useful for CI).
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
