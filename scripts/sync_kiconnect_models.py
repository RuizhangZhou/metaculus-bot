from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request


DEFAULT_CHAT_MODEL_PRIORITY = [
    "gpt-5.2",
    "gpt-5.4-mini",
    "gpt-oss-120b",
    "mistral-small-4-119b-2603",
]

DEFAULT_NON_CHAT_PATTERNS = [
    r"(^|[-_/])(embed|embedding|embeddings)([-_/]|$)",
    r"(^|[-_/])e5([-_/]|$)",
    r"(^|[-_/])bge([-_/]|$)",
    r"(^|[-_/])gte([-_/]|$)",
    r"(^|[-_/])rerank(er)?([-_/]|$)",
    r"(^|[-_/])(whisper|tts|stt|speech|transcri(pt|be))([-_/]|$)",
]

DEFAULT_GITHUB_TOKEN_ENV_NAMES = [
    "GITHUB_TOKEN",
    "GH_TOKEN",
    "GH_TOKEN_MetaculusBot",
]

TEMPLATE_BLOCK_START = "# BEGIN AUTO-KICONNECT-MODELS"
TEMPLATE_BLOCK_END = "# END AUTO-KICONNECT-MODELS"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch the current KIconnect model list, keep chat models only, "
            "and sync local/GitHub configuration."
        )
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Local dotenv file to update with KICONNECT_MODEL and KICONNECT_MODEL_FALLBACKS.",
    )
    parser.add_argument(
        "--env-template",
        default=".env.template",
        help="Template file where the managed KIconnect summary block should be updated.",
    )
    parser.add_argument(
        "--manifest",
        default=".github/kiconnect_models.json",
        help="JSON manifest path for the fetched and filtered model lists.",
    )
    parser.add_argument(
        "--github-repo",
        default="",
        help="GitHub repo slug (owner/repo). Defaults to GITHUB_REPOSITORY or git origin.",
    )
    parser.add_argument(
        "--update-github-vars",
        action="store_true",
        help="Also sync GitHub repository variables.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the derived values without writing files or GitHub variables.",
    )
    return parser.parse_args()


def load_dotenv_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def resolve_setting(name: str, env_file_values: dict[str, str]) -> str:
    return os.getenv(name, "").strip() or env_file_values.get(name, "").strip()


def fetch_available_models(api_url: str, api_key: str) -> list[dict[str, Any]]:
    url = f"{api_url.rstrip('/')}/models"
    req = request.Request(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        },
        method="GET",
    )
    with request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    data = payload.get("data")
    if not isinstance(data, list):
        raise ValueError("KIconnect /models response is missing a data list")
    return [item for item in data if isinstance(item, dict)]


def is_chat_model(model_id: str) -> bool:
    lowered = model_id.strip().lower()
    if not lowered:
        return False
    return not any(re.search(pattern, lowered) for pattern in DEFAULT_NON_CHAT_PATTERNS)


def filter_chat_models(model_ids: list[str]) -> list[str]:
    seen: set[str] = set()
    chat_models: list[str] = []
    for model_id in model_ids:
        cleaned = model_id.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        if is_chat_model(cleaned):
            chat_models.append(cleaned)
    return chat_models


def prioritize_chat_models(chat_models: list[str]) -> list[str]:
    priority_index = {
        model_id: index for index, model_id in enumerate(DEFAULT_CHAT_MODEL_PRIORITY)
    }
    return sorted(
        chat_models,
        key=lambda model_id: (
            0 if model_id in priority_index else 1,
            priority_index.get(model_id, len(DEFAULT_CHAT_MODEL_PRIORITY)),
            model_id,
        ),
    )


def choose_primary_and_fallbacks(chat_models: list[str]) -> tuple[str, list[str]]:
    ordered = prioritize_chat_models(chat_models)
    if not ordered:
        raise ValueError("No KIconnect chat models were detected")
    return ordered[0], ordered[1:]


def update_dotenv_file(path: Path, updates: dict[str, str], dry_run: bool) -> bool:
    if not path.exists():
        return False

    original_text = path.read_text(encoding="utf-8")
    newline = "\r\n" if "\r\n" in original_text else "\n"
    lines = original_text.splitlines()
    pending = dict(updates)
    new_lines: list[str] = []

    for line in lines:
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            new_lines.append(line)
            continue

        key, _ = line.split("=", 1)
        normalized_key = key.strip()
        if normalized_key in pending:
            new_lines.append(f"{normalized_key}={pending.pop(normalized_key)}")
        else:
            new_lines.append(line)

    for key, value in pending.items():
        new_lines.append(f"{key}={value}")

    updated_text = newline.join(new_lines)
    if original_text.endswith(("\n", "\r")):
        updated_text += newline

    if updated_text == original_text:
        return False

    if dry_run:
        return True

    path.write_text(updated_text, encoding="utf-8")
    return True


def build_template_block(chat_models: list[str], primary: str, fallbacks: list[str]) -> str:
    fallback_text = ",".join(fallbacks) if fallbacks else "(none)"
    lines = [
        TEMPLATE_BLOCK_START,
        "# This block is managed by scripts/sync_kiconnect_models.py",
        f"# Current KICONNECT chat models: {','.join(chat_models)}",
        f"# Recommended KICONNECT_MODEL: {primary}",
        f"# Recommended KICONNECT_MODEL_FALLBACKS: {fallback_text}",
        TEMPLATE_BLOCK_END,
    ]
    return "\n".join(lines)


def update_template_file(
    path: Path, chat_models: list[str], primary: str, fallbacks: list[str], dry_run: bool
) -> bool:
    if not path.exists():
        return False

    original_text = path.read_text(encoding="utf-8")
    newline = "\r\n" if "\r\n" in original_text else "\n"
    managed_block = build_template_block(chat_models, primary, fallbacks).replace(
        "\n", newline
    )

    if TEMPLATE_BLOCK_START in original_text and TEMPLATE_BLOCK_END in original_text:
        updated_text = re.sub(
            re.escape(TEMPLATE_BLOCK_START)
            + r".*?"
            + re.escape(TEMPLATE_BLOCK_END),
            managed_block,
            original_text,
            flags=re.DOTALL,
        )
    else:
        anchor = "# Optional: Use KIconnect as your non-search LLM (OpenAI-compatible proxy)"
        insertion = managed_block + newline
        if anchor in original_text:
            updated_text = original_text.replace(anchor, f"{anchor}{newline}{insertion}", 1)
        else:
            updated_text = insertion + original_text

    if updated_text == original_text:
        return False

    if dry_run:
        return True

    path.write_text(updated_text, encoding="utf-8")
    return True


def write_manifest(
    path: Path,
    available_models: list[dict[str, Any]],
    chat_models: list[str],
    primary: str,
    fallbacks: list[str],
    dry_run: bool,
) -> bool:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "available_models": [item.get("id", "") for item in available_models],
        "chat_models": chat_models,
        "primary_model": primary,
        "fallback_models": fallbacks,
    }
    updated_text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    original_text = path.read_text(encoding="utf-8") if path.exists() else ""

    if updated_text == original_text:
        return False

    if dry_run:
        return True

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(updated_text, encoding="utf-8")
    return True


def read_github_token_from_known_sources() -> str:
    for env_name in DEFAULT_GITHUB_TOKEN_ENV_NAMES:
        token = os.getenv(env_name, "").strip()
        if token:
            return token

    openclaw_env = Path.home() / ".openclaw" / ".env"
    openclaw_values = load_dotenv_file(openclaw_env)
    for env_name in DEFAULT_GITHUB_TOKEN_ENV_NAMES:
        token = openclaw_values.get(env_name, "").strip()
        if token:
            return token

    try:
        result = subprocess.run(
            ["gh", "auth", "token"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return ""
    return result.stdout.strip()


def infer_github_repo(explicit_repo: str) -> str:
    if explicit_repo:
        return explicit_repo

    env_repo = os.getenv("GITHUB_REPOSITORY", "").strip()
    if env_repo:
        return env_repo

    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return ""

    remote = result.stdout.strip()
    match = re.search(r"github\.com[:/](?P<repo>[^/]+/[^/.]+)(?:\.git)?$", remote)
    return match.group("repo") if match else ""


def github_api_request(
    repo: str, token: str, method: str, endpoint: str, payload: dict[str, str]
) -> None:
    url = f"https://api.github.com/repos/{repo}{endpoint}"
    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        method=method,
    )
    with request.urlopen(req, timeout=30):
        return None


def upsert_github_variable(repo: str, token: str, name: str, value: str) -> None:
    payload = {"name": name, "value": value}
    try:
        github_api_request(
            repo,
            token,
            "PATCH",
            f"/actions/variables/{name}",
            payload,
        )
    except error.HTTPError as exc:
        if exc.code != 404:
            raise
        github_api_request(repo, token, "POST", "/actions/variables", payload)


def main() -> int:
    args = parse_args()
    env_path = Path(args.env_file)
    env_template_path = Path(args.env_template)
    manifest_path = Path(args.manifest)

    env_values = load_dotenv_file(env_path)
    api_url = resolve_setting("KICONNECT_API_URL", env_values)
    api_key = resolve_setting("KICONNECT_API_KEY", env_values)
    if not api_url or not api_key:
        raise SystemExit(
            "KICONNECT_API_URL and KICONNECT_API_KEY must be available via environment or the env file."
        )

    available_models = fetch_available_models(api_url, api_key)
    available_model_ids = [
        str(item.get("id", "")).strip()
        for item in available_models
        if str(item.get("id", "")).strip()
    ]
    chat_models = prioritize_chat_models(filter_chat_models(available_model_ids))
    primary, fallbacks = choose_primary_and_fallbacks(chat_models)
    fallback_value = ",".join(fallbacks)

    print("Available models:", ",".join(available_model_ids))
    print("Chat models:", ",".join(chat_models))
    print("Primary model:", primary)
    print("Fallback models:", fallback_value or "(none)")

    env_changed = update_dotenv_file(
        env_path,
        {
            "KICONNECT_MODEL": primary,
            "KICONNECT_MODEL_FALLBACKS": fallback_value,
        },
        args.dry_run,
    )
    template_changed = update_template_file(
        env_template_path, chat_models, primary, fallbacks, args.dry_run
    )
    manifest_changed = write_manifest(
        manifest_path, available_models, chat_models, primary, fallbacks, args.dry_run
    )

    if env_changed:
        print(f"Updated {env_path}")
    if template_changed:
        print(f"Updated {env_template_path}")
    if manifest_changed:
        print(f"Updated {manifest_path}")

    if args.update_github_vars:
        repo = infer_github_repo(args.github_repo)
        if not repo:
            raise SystemExit(
                "Could not infer GitHub repository. Pass --github-repo owner/repo."
            )
        token = read_github_token_from_known_sources()
        if not token:
            raise SystemExit(
                "No GitHub token found. Set GITHUB_TOKEN/GH_TOKEN or ensure gh is logged in."
            )

        if args.dry_run:
            print(
                f"Would update GitHub repo variables for {repo}: "
                "KICONNECT_MODEL, KICONNECT_MODEL_FALLBACKS, KICONNECT_CHAT_MODELS"
            )
        else:
            upsert_github_variable(repo, token, "KICONNECT_MODEL", primary)
            upsert_github_variable(
                repo, token, "KICONNECT_MODEL_FALLBACKS", fallback_value
            )
            upsert_github_variable(
                repo, token, "KICONNECT_CHAT_MODELS", ",".join(chat_models)
            )
            print(f"Updated GitHub repo variables for {repo}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
