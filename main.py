import argparse
import asyncio
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import dotenv
import requests

from digest_mode import load_tournament_identifiers, matrix_send_message, run_digest
from forecasting_tools import GeneralLlm, MetaculusClient

from template_bot_2026 import SpringTemplateBot2026


def _notify_matrix_on_submit(
    *, run_mode: str, forecast_reports: list[object]
) -> None:
    if run_mode not in {"tournament", "metaculus_cup"}:
        return
    successful_reports = [
        report
        for report in forecast_reports
        if not isinstance(report, BaseException)
    ]
    failed_reports = [
        report for report in forecast_reports if isinstance(report, BaseException)
    ]

    notify_always = (
        os.getenv("MATRIX_NOTIFY_ALWAYS", "").strip().lower()
        in {"1", "true", "yes", "y"}
    )
    if not notify_always and not successful_reports and not failed_reports:
        return

    now_iso = datetime.now(timezone.utc).isoformat()
    lines = [
        f"Metaculus bot run ({now_iso[:10]}): {len(successful_reports)} submitted, {len(failed_reports)} failed"
    ]
    for report in successful_reports[:10]:
        question = getattr(report, "question", None)
        if question is None:
            continue
        question_text = getattr(question, "question_text", "(unknown question)")
        page_url = getattr(question, "page_url", None) or f"post:{getattr(question, 'id_of_post', '')}"
        lines.append(f"- {question_text} ({page_url})")
    if len(successful_reports) > 10:
        lines.append(f"... and {len(successful_reports) - 10} more")
    if failed_reports:
        lines.append("")
        lines.append("Failures (first 5):")
        for err in failed_reports[:5]:
            lines.append(f"- {repr(err)[:300]}")
    matrix_send_message("\n".join(lines))


def _validate_and_normalize_metaculus_token() -> None:
    token_raw = os.getenv("METACULUS_TOKEN", "")
    token = token_raw.strip()
    if token != token_raw:
        os.environ["METACULUS_TOKEN"] = token

    if not token:
        message = (
            "METACULUS_TOKEN is missing. Set it as a GitHub Actions Repository secret, "
            "or (if you stored it as an Environment secret) add `environment: <name>` to the workflow job."
        )
        matrix_send_message(f"Metaculus bot error: {message}")
        raise SystemExit(message)

    lowered = token.lower()
    if lowered.startswith("token ") or lowered.startswith("bearer "):
        message = (
            "METACULUS_TOKEN should be the raw token value (no 'Token ' / 'Bearer ' prefix). "
            "Update the secret/env var to only the token string."
        )
        matrix_send_message(f"Metaculus bot error: {message}")
        raise SystemExit(message)

    if any(ch.isspace() for ch in token):
        message = (
            "METACULUS_TOKEN contains whitespace. Re-copy the token (no spaces/newlines) and update the secret."
        )
        matrix_send_message(f"Metaculus bot error: {message}")
        raise SystemExit(message)


if __name__ == "__main__":
    dotenv.load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False

    parser = argparse.ArgumentParser(
        description="Run the TemplateBot forecasting system"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "metaculus_cup", "test_questions", "digest"],
        default="tournament",
        help="Specify the run mode (default: tournament)",
    )
    parser.add_argument(
        "--tournaments-file",
        type=str,
        default=None,
        help="Path to a file containing tournament URLs/slugs (one per line).",
    )
    parser.add_argument(
        "--tournament",
        action="append",
        default=[],
        help="Tournament slug/id or URL (repeatable). Extends --tournaments-file.",
    )
    parser.add_argument(
        "--researcher",
        type=str,
        default=None,
        help=(
            "Override research strategy/model. Examples: "
            "'no_research', 'asknews/news-summaries', "
            "'asknews/deep-research/low-depth', 'smart-searcher/<model>'."
        ),
    )
    parser.add_argument(
        "--default-model",
        type=str,
        default=None,
        help=(
            "Override the default LLM model (Litellm format), e.g. "
            "'openrouter/openai/gpt-oss-120b:free' or 'openrouter/openai/gpt-4o'."
        ),
    )
    parser.add_argument(
        "--parser-model",
        type=str,
        default=None,
        help=(
            "Override the parser LLM model used for structured output, e.g. "
            "'openrouter/openai/gpt-oss-120b:free'."
        ),
    )
    parser.add_argument(
        "--submit",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Submit forecasts to Metaculus. Defaults: tournament/metaculus_cup=True, "
            "test_questions=False, digest=False."
        ),
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "metaculus_cup", "test_questions", "digest"] = (
        args.mode
    )

    _validate_and_normalize_metaculus_token()

    default_submit = run_mode in {"tournament", "metaculus_cup"}
    if run_mode == "test_questions":
        default_submit = False
    publish_to_metaculus = default_submit if args.submit is None else bool(args.submit)
    if run_mode == "digest" and publish_to_metaculus:
        logging.getLogger(__name__).warning(
            "--submit ignored in digest mode (no submission)."
        )
        publish_to_metaculus = False

    llms: dict[str, object] | None = None
    if args.researcher or args.default_model or args.parser_model:
        llms = {}
    if args.researcher and llms is not None:
        llms["researcher"] = args.researcher
    if args.default_model and llms is not None:
        llms["default"] = GeneralLlm(
            model=args.default_model,
            temperature=0.3,
            timeout=60,
            allowed_tries=2,
        )
    if args.parser_model and llms is not None:
        llms["parser"] = GeneralLlm(
            model=args.parser_model,
            temperature=0.0,
            timeout=60,
            allowed_tries=2,
        )

    template_bot = SpringTemplateBot2026(
        research_reports_per_question=1,
        predictions_per_research_report=5,
        use_research_summary_to_forecast=False,
        llms=llms,
        publish_reports_to_metaculus=publish_to_metaculus,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=(run_mode == "tournament"),
        extra_metadata_in_explanation=True,
    )

    client = MetaculusClient()
    forecast_reports: list[object] = []

    try:
        if run_mode == "tournament":
            if args.tournaments_file or args.tournament:
                tournaments, unsupported = load_tournament_identifiers(
                    args.tournaments_file, args.tournament
                )
                for item in unsupported:
                    logging.getLogger(__name__).warning(
                        f"Unsupported collection (not a tournament): {item}"
                    )
                if not tournaments:
                    raise SystemExit(
                        "No valid tournaments configured via --tournaments-file/--tournament."
                    )
                for tournament_id in tournaments:
                    try:
                        forecast_reports.extend(
                            asyncio.run(
                                template_bot.forecast_on_tournament(
                                    tournament_id, return_exceptions=True
                                )
                            )
                        )
                    except Exception as e:
                        logging.getLogger(__name__).error(
                            f"Failed to forecast on tournament '{tournament_id}': {e}"
                        )
                        forecast_reports.append(e)
            else:
                seasonal_tournament_reports = asyncio.run(
                    template_bot.forecast_on_tournament(
                        client.CURRENT_AI_COMPETITION_ID, return_exceptions=True
                    )
                )
                minibench_reports = asyncio.run(
                    template_bot.forecast_on_tournament(
                        client.CURRENT_MINIBENCH_ID, return_exceptions=True
                    )
                )
                forecast_reports = seasonal_tournament_reports + minibench_reports

        elif run_mode == "metaculus_cup":
            template_bot.skip_previously_forecasted_questions = False
            forecast_reports = asyncio.run(
                template_bot.forecast_on_tournament(
                    client.CURRENT_METACULUS_CUP_ID, return_exceptions=True
                )
            )

        elif run_mode == "test_questions":
            EXAMPLE_QUESTIONS = [
                "https://www.metaculus.com/questions/578/human-extinction-by-2100/",
                "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",
                "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",
                "https://www.metaculus.com/c/diffusion-community/38880/how-many-us-labor-strikes-due-to-ai-in-2029/",
            ]
            template_bot.skip_previously_forecasted_questions = False
            questions = [
                client.get_question_by_url(question_url)
                for question_url in EXAMPLE_QUESTIONS
            ]
            forecast_reports = asyncio.run(
                template_bot.forecast_questions(questions, return_exceptions=True)
            )

        elif run_mode == "digest":
            tournaments_file = args.tournaments_file or "tracked_tournaments.txt"
            tournaments, unsupported = load_tournament_identifiers(
                tournaments_file, args.tournament
            )
            for item in unsupported:
                logging.getLogger(__name__).warning(
                    f"Unsupported collection (not a tournament): {item}"
                )
            if not tournaments:
                raise SystemExit(
                    "No tournaments configured. Add URLs/slugs to tracked_tournaments.txt or pass --tournament."
                )

            template_bot.publish_reports_to_metaculus = False
            template_bot.skip_previously_forecasted_questions = False
            state_path = Path(".state") / "digest_state.json"
            out_dir = Path("reports") / "digest"
            asyncio.run(
                run_digest(
                    template_bot=template_bot,
                    tournaments=tournaments,
                    state_path=state_path,
                    out_dir=out_dir,
                )
            )
            forecast_reports = []
    except requests.exceptions.HTTPError as e:
        if "Invalid token" in str(e):
            message = (
                "Metaculus API rejected your METACULUS_TOKEN (Invalid token). "
                "Double-check you copied the token from Metaculus account settings, "
                "and that GitHub Actions is actually receiving it (Repository secret vs Environment secret)."
            )
            matrix_send_message(f"Metaculus bot error: {message}")
            raise SystemExit(message) from e
        raise

    template_bot.log_report_summary(forecast_reports)  # type: ignore[arg-type]
    if publish_to_metaculus:
        _notify_matrix_on_submit(run_mode=run_mode, forecast_reports=forecast_reports)
