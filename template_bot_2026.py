"""
Legacy implementation (was `main.py`).
Use `main.py` as the entrypoint.
"""

import argparse
import asyncio
import logging
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import dotenv
from typing import Literal

import requests

from forecasting_tools import (
    AskNewsSearcher,
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusClient,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    DateQuestion,
    DatePercentile,
    Percentile,
    ConditionalQuestion,
    ConditionalPrediction,
    PredictionTypes,
    PredictionAffirmed,
    BinaryPrediction,
    PredictedOptionList,
    ReasonedPrediction,
    SmartSearcher,
    clean_indents,
    structure_output,
)

dotenv.load_dotenv()
logger = logging.getLogger(__name__)


class SpringTemplateBot2026(ForecastBot):
    """
    This is the template bot for Spring 2026 Metaculus AI Tournament.
    This is a copy of what is used by Metaculus to run the Metac Bots in our benchmark, provided as a template for new bot makers.
    This template is given as-is, and is use-at-your-own-risk.
    We have covered most test cases in forecasting-tools it may be worth double checking key components locally.
    So far our track record has been 1 mentionable bug per season (affecting forecasts for 1-2% of total questions)

    Main changes since Fall:
    - Additional prompting has been added to numeric questions to emphasize putting pecentile values in the correct order.
    - Support for conditional and date questions has been added
    - Note: Spring AIB will not use date/conditional questions, so these are only for forecasting on the main site as you wish.

    The main entry point of this bot is `bot.forecast_on_tournament(tournament_id)` in the parent class.
    See the script at the bottom of the file for more details on how to run the bot.
    Ignoring the finer details, the general flow is:
    - Load questions from Metaculus
    - For each question
        - Execute run_research a number of times equal to research_reports_per_question
        - Execute respective run_forecast function `predictions_per_research_report * research_reports_per_question` times
        - Aggregate the predictions
        - Submit prediction (if publish_reports_to_metaculus is True)
    - Return a list of ForecastReport objects

    Alternatively, you can use the MetaculusClient to make a custom filter of questions to forecast on
    and forecast them with `bot.forecast_questions(questions)`

    Only the research and forecast functions need to be implemented in ForecastBot subclasses,
    though you may want to override other ForecastBot functions.
    In this example, you can change the prompts to be whatever you want since,
    structure_output uses an LLM to intelligently reformat the output into the needed structure.

    By default (i.e. 'tournament' mode), when you run this script, it will forecast on any open questions in the
    primary bot tournament and MiniBench. If you want to forecast on only one or the other, you can remove one
    of them from the 'tournament' mode code at the bottom of the file.

    You can experiment with what models work best with your bot by using the `llms` parameter when initializing the bot.
    You can initialize the bot with any number of models. For example,
    ```python
    my_bot = MyBot(
        ...
        llms={  # choose your model names or GeneralLlm llms here, otherwise defaults will be chosen for you
            "default": GeneralLlm(
                model="openrouter/openai/gpt-4o", # "anthropic/claude-sonnet-4-20250514", etc (see docs for litellm)
                temperature=0.3,
                timeout=40,
                allowed_tries=2,
            ),
            "summarizer": "openai/gpt-4o-mini",
            "researcher": "asknews/news-summaries",
            "parser": "openai/gpt-4o-mini",
        },
    )
    ```

    Then you can access the model in custom functions like this:
    ```python
    research_strategy = self.get_llm("researcher", "model_name"
    if research_strategy == "asknews/news-summaries":
        ...
    # OR
    summarizer = await self.get_llm("summarizer", "llm").invoke(prompt)
    # OR
    reasoning = await self.get_llm("default", "llm").invoke(prompt)
    ```

    If you end up having trouble with rate limits and want to try a more sophisticated rate limiter try:
    ```python
    from forecasting_tools import RefreshingBucketRateLimiter
    rate_limiter = RefreshingBucketRateLimiter(
        capacity=2,
        refresh_rate=1,
    ) # Allows 1 request per second on average with a burst of 2 requests initially. Set this as a class variable
    await self.rate_limiter.wait_till_able_to_acquire_resources(1) # 1 because it's consuming 1 request (use more if you are adding a token limit)
    ```
    Additionally OpenRouter has large rate limits immediately on account creation
    """

    _max_concurrent_questions = (
        1  # Set this to whatever works for your search-provider/ai-model rate limits
    )
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)
    _structure_output_validation_samples = 2

    ##################################### RESEARCH #####################################

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            research = ""
            researcher = self.get_llm("researcher")

            prompt = clean_indents(
                f"""
                You are an assistant to a superforecaster.
                The superforecaster will give you a question they intend to forecast on.
                To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.
                You do not produce forecasts yourself.

                Question:
                {question.question_text}

                This question's outcome will be determined by the specific criteria below:
                {question.resolution_criteria}

                {question.fine_print}
                """
            )

            if isinstance(researcher, GeneralLlm):
                research = await researcher.invoke(prompt)
            elif (
                researcher == "asknews/news-summaries"
                or researcher == "asknews/deep-research/low-depth"
                or researcher == "asknews/deep-research/medium-depth"
                or researcher == "asknews/deep-research/high-depth"
            ):
                try:
                    research = await AskNewsSearcher().call_preconfigured_version(
                        researcher, prompt
                    )
                except Exception as e:
                    error_text = str(e)
                    if "reserved for Spelunker and Analyst tiers" in error_text:
                        logger.warning(
                            "AskNews API access is not enabled for this plan (403). "
                            "Disable AskNews research or upgrade your AskNews plan. "
                            f"Error: {error_text}"
                        )
                    else:
                        logger.warning(f"AskNews research failed, continuing without it: {error_text}")
                    research = ""
            elif researcher.startswith("smart-searcher"):
                model_name = researcher.removeprefix("smart-searcher/")
                try:
                    num_searches_to_run = int(
                        os.getenv("SMART_SEARCHER_NUM_SEARCHES", "2")
                    )
                except Exception:
                    num_searches_to_run = 2
                try:
                    num_sites_per_search = int(
                        os.getenv("SMART_SEARCHER_NUM_SITES_PER_SEARCH", "10")
                    )
                except Exception:
                    num_sites_per_search = 10
                use_advanced_filters = (
                    os.getenv("SMART_SEARCHER_USE_ADVANCED_FILTERS", "")
                    .strip()
                    .lower()
                    in {"1", "true", "yes", "y"}
                )
                searcher = SmartSearcher(
                    model=model_name,
                    temperature=0,
                    num_searches_to_run=max(1, num_searches_to_run),
                    num_sites_per_search=max(1, num_sites_per_search),
                    use_advanced_filters=use_advanced_filters,
                )
                research = await searcher.invoke(prompt)
            elif not researcher or researcher == "None" or researcher == "no_research":
                research = ""
            else:
                research = await self.get_llm("researcher", "llm").invoke(prompt)
            logger.info(f"Found Research for URL {question.page_url}:\n{research}")
            return research

    ##################################### BINARY QUESTIONS #####################################

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Question background:
            {question.background_info}


            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A brief description of a scenario that results in a No outcome.
            (d) A brief description of a scenario that results in a Yes outcome.

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.
            {self._get_conditional_disclaimer_if_necessary(question)}

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )

        return await self._binary_prompt_to_forecast(question, prompt)

    async def _binary_prompt_to_forecast(
        self,
        question: BinaryQuestion,
        prompt: str,
    ) -> ReasonedPrediction[float]:
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        binary_prediction: BinaryPrediction = await structure_output(
            reasoning,
            BinaryPrediction,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
        )
        decimal_pred = max(0.01, min(0.99, binary_prediction.prediction_in_decimal))

        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {decimal_pred}."
        )
        return ReasonedPrediction(prediction_value=decimal_pred, reasoning=reasoning)

    ##################################### MULTIPLE CHOICE QUESTIONS #####################################

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            The options are: {question.options}


            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A description of an scenario that results in an unexpected outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}
            You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.

            The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )
        return await self._multiple_choice_prompt_to_forecast(question, prompt)

    async def _multiple_choice_prompt_to_forecast(
        self,
        question: MultipleChoiceQuestion,
        prompt: str,
    ) -> ReasonedPrediction[PredictedOptionList]:
        parsing_instructions = clean_indents(
            f"""
            Make sure that all option names are one of the following:
            {question.options}

            The text you are parsing may prepend these options with some variation of "Option" which you should remove if not part of the option names I just gave you.
            Additionally, you may sometimes need to parse a 0% probability. Please do not skip options with 0% but rather make it an entry in your final list with 0% probability.
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        predicted_option_list: PredictedOptionList = await structure_output(
            text_to_structure=reasoning,
            output_type=PredictedOptionList,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
            additional_instructions=parsing_instructions,
        )

        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {predicted_option_list}."
        )
        return ReasonedPrediction(
            prediction_value=predicted_option_list, reasoning=reasoning
        )

    ##################################### NUMERIC QUESTIONS #####################################

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Units for answer: {question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - Please notice the units requested and give your answer in these units (e.g. whether you represent a number as 1,000,000 or 1 million).
            - Never use scientific notation.
            - Always start with a smaller number (more negative if negative) and then increase from there. The value for percentile 10 should always be less than the value for percentile 20, and so on.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}
            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            The last thing you write is your final answer as:
            "
            Percentile 10: XX (lowest number value)
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX (highest number value)
            "
            """
        )
        return await self._numeric_prompt_to_forecast(question, prompt)

    async def _numeric_prompt_to_forecast(
        self,
        question: NumericQuestion,
        prompt: str,
    ) -> ReasonedPrediction[NumericDistribution]:
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        parsing_instructions = clean_indents(
            f"""
            The text given to you is trying to give a forecast distribution for a numeric question.
            - This text is trying to answer the numeric question: "{question.question_text}".
            - When parsing the text, please make sure to give the values (the ones assigned to percentiles) in terms of the correct units.
            - The units for the forecast are: {question.unit_of_measure}
            - Your work will be shown publicly with these units stated verbatim after the numbers your parse.
            - As an example, someone else guessed that the answer will be between {question.lower_bound} {question.unit_of_measure} and {question.upper_bound} {question.unit_of_measure}, so the numbers parsed from an answer like this would be verbatim "{question.lower_bound}" and "{question.upper_bound}".
            - If the answer doesn't give the answer in the correct units, you should parse it in the right units. For instance if the answer gives numbers as $500,000,000 and units are "B $" then you should parse the answer as 0.5 (since $500,000,000 is $0.5 billion).
            - If percentiles are not explicitly given (e.g. only a single value is given) please don't return a parsed output, but rather indicate that the answer is not explicitly given in the text.
            - Turn any values that are in scientific notation into regular numbers.
            """
        )
        percentile_list: list[Percentile] = await structure_output(
            reasoning,
            list[Percentile],
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
            num_validation_samples=self._structure_output_validation_samples,
        )
        prediction = NumericDistribution.from_question(percentile_list, question)
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {prediction.declared_percentiles}."
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    ##################################### DATE QUESTIONS #####################################

    async def _run_forecast_on_date(
        self, question: DateQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - This is a date question, and as such, the answer must be expressed in terms of dates.
            - The dates must be written in the format of YYYY-MM-DD. If hours matter, please append the date with the hour in UTC and military time: YYYY-MM-DDTHH:MM:SSZ.No other formatting is allowed.
            - Always start with a lower date chronologically and then increase from there.
            - Do NOT forget this. The dates must be written in chronological order starting at the earliest time at percentile 10 and increasing from there.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}
            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            The last thing you write is your final answer as:
            "
            Percentile 10: YYYY-MM-DD (oldest date)
            Percentile 20: YYYY-MM-DD
            Percentile 40: YYYY-MM-DD
            Percentile 60: YYYY-MM-DD
            Percentile 80: YYYY-MM-DD
            Percentile 90: YYYY-MM-DD (newest date)
            "
            """
        )
        forecast = await self._date_prompt_to_forecast(question, prompt)
        return forecast

    async def _date_prompt_to_forecast(
        self,
        question: DateQuestion,
        prompt: str,
    ) -> ReasonedPrediction[NumericDistribution]:
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        parsing_instructions = clean_indents(
            f"""
            The text given to you is trying to give a forecast distribution for a date question.
            - This text is trying to answer the question: "{question.question_text}".
            - As an example, someone else guessed that the answer will be between {question.lower_bound} and {question.upper_bound}, so the numbers parsed from an answer like this would be verbatim "{question.lower_bound}" and "{question.upper_bound}".
            - The output is given as dates/times please format it into a valid datetime parsable string. Assume midnight UTC if no hour is given.
            - If percentiles are not explicitly given (e.g. only a single value is given) please don't return a parsed output, but rather indicate that the answer is not explicitly given in the text.
            """
        )
        date_percentile_list: list[DatePercentile] = await structure_output(
            reasoning,
            list[DatePercentile],
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
            num_validation_samples=self._structure_output_validation_samples,
        )

        percentile_list = [
            Percentile(
                percentile=percentile.percentile,
                value=percentile.value.timestamp(),
            )
            for percentile in date_percentile_list
        ]
        prediction = NumericDistribution.from_question(percentile_list, question)
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {prediction.declared_percentiles}."
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    def _create_upper_and_lower_bound_messages(
        self, question: NumericQuestion | DateQuestion
    ) -> tuple[str, str]:
        if isinstance(question, NumericQuestion):
            if question.nominal_upper_bound is not None:
                upper_bound_number = question.nominal_upper_bound
            else:
                upper_bound_number = question.upper_bound
            if question.nominal_lower_bound is not None:
                lower_bound_number = question.nominal_lower_bound
            else:
                lower_bound_number = question.lower_bound
            unit_of_measure = question.unit_of_measure
        elif isinstance(question, DateQuestion):
            upper_bound_number = question.upper_bound.date().isoformat()
            lower_bound_number = question.lower_bound.date().isoformat()
            unit_of_measure = ""
        else:
            raise ValueError()

        if question.open_upper_bound:
            upper_bound_message = f"The question creator thinks the number is likely not higher than {upper_bound_number} {unit_of_measure}."
        else:
            upper_bound_message = f"The outcome can not be higher than {upper_bound_number} {unit_of_measure}."

        if question.open_lower_bound:
            lower_bound_message = f"The question creator thinks the number is likely not lower than {lower_bound_number} {unit_of_measure}."
        else:
            lower_bound_message = f"The outcome can not be lower than {lower_bound_number} {unit_of_measure}."
        return upper_bound_message, lower_bound_message

    ##################################### CONDITIONAL QUESTIONS #####################################

    async def _run_forecast_on_conditional(
        self, question: ConditionalQuestion, research: str
    ) -> ReasonedPrediction[ConditionalPrediction]:
        parent_info, full_research = await self._get_question_prediction_info(
            question.parent, research, "parent"
        )
        child_info, full_research = await self._get_question_prediction_info(
            question.child, research, "child"
        )
        yes_info, full_research = await self._get_question_prediction_info(
            question.question_yes, full_research, "yes"
        )
        no_info, full_research = await self._get_question_prediction_info(
            question.question_no, full_research, "no"
        )
        full_reasoning = clean_indents(
            f"""
            ## Parent Question Reasoning
            {parent_info.reasoning}
            ## Child Question Reasoning
            {child_info.reasoning}
            ## Yes Question Reasoning
            {yes_info.reasoning}
            ## No Question Reasoning
            {no_info.reasoning}
        """
        )
        full_prediction = ConditionalPrediction(
            parent=parent_info.prediction_value,  # type: ignore
            child=child_info.prediction_value,  # type: ignore
            prediction_yes=yes_info.prediction_value,  # type: ignore
            prediction_no=no_info.prediction_value,  # type: ignore
        )
        return ReasonedPrediction(
            reasoning=full_reasoning, prediction_value=full_prediction
        )

    async def _get_question_prediction_info(
        self, question: MetaculusQuestion, research: str, question_type: str
    ) -> tuple[ReasonedPrediction[PredictionTypes | PredictionAffirmed], str]:
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        previous_forecasts = question.previous_forecasts
        if (
            question_type in ["parent", "child"]
            and previous_forecasts
            and question_type not in self.force_reforecast_in_conditional
        ):
            # TODO: add option to not affirm current parent/child forecasts, create new forecast
            previous_forecast = previous_forecasts[-1]
            current_utc_time = datetime.now(timezone.utc)
            if (
                previous_forecast.timestamp_end is None
                or previous_forecast.timestamp_end > current_utc_time
            ):
                pretty_value = DataOrganizer.get_readable_prediction(previous_forecast)  # type: ignore
                prediction = ReasonedPrediction(
                    prediction_value=PredictionAffirmed(),
                    reasoning=f"Already existing forecast reaffirmed at {pretty_value}.",
                )
                return (prediction, research)  # type: ignore
        info = await self._make_prediction(question, research)
        full_research = self._add_reasoning_to_research(research, info, question_type)
        return info, full_research  # type: ignore

    def _add_reasoning_to_research(
        self,
        research: str,
        reasoning: ReasonedPrediction[PredictionTypes],
        question_type: str,
    ) -> str:
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        question_type = question_type.title()
        return clean_indents(
            f"""
            {research}
            ---
            ## {question_type} Question Information
            You have previously forecasted the {question_type} Question to the value: {DataOrganizer.get_readable_prediction(reasoning.prediction_value)}
            This is relevant information for your current forecast, but it is NOT your current forecast, but previous forecasting information that is relevant to your current forecast.
            The reasoning for the {question_type} Question was as such:
            ```
            {reasoning.reasoning}
            ```
            This is absolutely essential: do NOT use this reasoning to re-forecast the {question_type} question.
            """
        )

    def _get_conditional_disclaimer_if_necessary(
        self, question: MetaculusQuestion
    ) -> str:
        if question.conditional_type not in ["yes", "no"]:
            return ""
        return clean_indents(
            """
            As you are given a conditional question with a parent and child, you are to only forecast the **CHILD** question, given the parent question's resolution.
            You never re-forecast the parent question under any circumstances, but you use probabilistic reasoning, strongly considering the parent question's resolution, to forecast the child question.
            """
        )


def _extract_tournament_identifier(value: str) -> str | None:
    raw = value.strip()
    if not raw or raw.startswith("#"):
        return None

    if raw.startswith("http://") or raw.startswith("https://"):
        tournament_match = re.search(r"/tournament/([^/]+)/?", raw)
        if tournament_match:
            slug = tournament_match.group(1).strip("/")
            return slug if slug.isdigit() else slug.lower()
        index_match = re.search(r"/index/([^/]+)/?", raw)
        if index_match:
            # Not supported by MetaculusClient tournament APIs; keep as sentinel so we can warn.
            return f"index:{index_match.group(1)}"
        return None

    slug = raw.strip("/")
    return slug if slug.isdigit() else slug.lower()


def _load_tournament_identifiers(
    tournaments_file: str | None, extra_identifiers: list[str] | None
) -> tuple[list[str], list[str]]:
    identifiers: list[str] = []
    unsupported: list[str] = []

    if tournaments_file:
        path = Path(tournaments_file)
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                identifier = _extract_tournament_identifier(line)
                if not identifier:
                    continue
                if identifier.startswith("index:"):
                    unsupported.append(identifier)
                else:
                    identifiers.append(identifier)
        else:
            logger.warning(f"Tournaments file not found: {tournaments_file}")

    for raw in extra_identifiers or []:
        identifier = _extract_tournament_identifier(raw)
        if not identifier:
            continue
        if identifier.startswith("index:"):
            unsupported.append(identifier)
        else:
            identifiers.append(identifier)

    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique_identifiers: list[str] = []
    for identifier in identifiers:
        if identifier in seen:
            continue
        seen.add(identifier)
        unique_identifiers.append(identifier)

    return unique_identifiers, unsupported


def _prediction_to_compact_jsonable(prediction: object) -> object:
    if isinstance(prediction, (float, int, str, bool)) or prediction is None:
        return prediction
    model_dump = getattr(prediction, "model_dump", None)
    if callable(model_dump):
        return model_dump()
    to_dict = getattr(prediction, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    as_dict = getattr(prediction, "dict", None)
    if callable(as_dict):
        return as_dict()
    return str(prediction)


def _get_close_time_iso(question: MetaculusQuestion) -> str | None:
    close_time = getattr(question, "close_time", None)
    if close_time is None:
        return None
    if isinstance(close_time, datetime):
        return close_time.astimezone(timezone.utc).isoformat()
    return str(close_time)


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _days_until(close_time_iso: str | None) -> float | None:
    dt = _parse_iso_datetime(close_time_iso)
    if dt is None:
        return None
    return (dt - datetime.now(timezone.utc)).total_seconds() / 86400


def _threshold_factor_from_days_left(days_left: float | None) -> float:
    if days_left is None:
        return 1.0
    if days_left <= 2:
        return 0.5
    if days_left <= 7:
        return 0.7
    return 1.0


def _get_numeric_percentile_map(pred: dict) -> dict[float, float]:
    percentiles = pred.get("declared_percentiles")
    if not isinstance(percentiles, list):
        return {}
    out: dict[float, float] = {}
    for p in percentiles:
        if not isinstance(p, dict):
            continue
        perc = p.get("percentile")
        val = p.get("value")
        if isinstance(perc, (int, float)) and isinstance(val, (int, float)):
            out[float(perc)] = float(val)
    return out


def _approx_median_from_percentiles(pmap: dict[float, float]) -> float | None:
    if 0.5 in pmap:
        return pmap[0.5]
    p40 = pmap.get(0.4)
    p60 = pmap.get(0.6)
    if p40 is not None and p60 is not None:
        return 0.5 * (p40 + p60)
    # Fallback: nearest available percentile to 0.5
    if not pmap:
        return None
    nearest = min(pmap.keys(), key=lambda p: abs(p - 0.5))
    return pmap[nearest]


def _is_significant_change(
    *,
    old_pred: object | None,
    new_pred: object | None,
    question_type: str,
    close_time_iso: str | None,
    old_close_time_iso: str | None,
) -> tuple[bool, str]:
    days_left = _days_until(close_time_iso)
    factor = _threshold_factor_from_days_left(days_left)

    if old_pred is None and new_pred is not None:
        return True, "new_question"
    if old_pred is not None and new_pred is None:
        return False, "question_missing_now"

    if close_time_iso and old_close_time_iso and close_time_iso != old_close_time_iso:
        new_dt = _parse_iso_datetime(close_time_iso)
        old_dt = _parse_iso_datetime(old_close_time_iso)
        if new_dt and old_dt:
            delta_hours = abs((new_dt - old_dt).total_seconds()) / 3600
            if delta_hours >= 24:
                return True, f"close_time_changed_{delta_hours:.1f}h"

    if question_type == "binary":
        if not isinstance(old_pred, (int, float)) or not isinstance(new_pred, (int, float)):
            return True, "binary_type_changed"
        old_p = float(old_pred)
        new_p = float(new_pred)
        base = 0.10
        threshold = base * factor
        abs_delta = abs(new_p - old_p)
        crossed = (old_p - 0.5) * (new_p - 0.5) < 0
        if abs_delta >= threshold:
            return True, f"abs_delta={abs_delta:.3f}"
        if crossed and abs_delta >= max(0.04, 0.5 * threshold):
            return True, f"crossed_50_abs_delta={abs_delta:.3f}"
        return False, f"abs_delta={abs_delta:.3f}"

    if question_type == "multiple_choice":
        if not isinstance(old_pred, dict) or not isinstance(new_pred, dict):
            return True, "mc_type_changed"
        # stored as {"predicted_options": [...]} or {"option": prob, ...}
        def to_map(d: dict) -> dict[str, float]:
            if "predicted_options" in d and isinstance(d["predicted_options"], list):
                m: dict[str, float] = {}
                for item in d["predicted_options"]:
                    if not isinstance(item, dict):
                        continue
                    name = item.get("option_name")
                    prob = item.get("probability")
                    if isinstance(name, str) and isinstance(prob, (int, float)):
                        m[name] = float(prob)
                return m
            return {
                k: float(v)
                for k, v in d.items()
                if isinstance(k, str) and isinstance(v, (int, float))
            }

        old_map = to_map(old_pred)
        new_map = to_map(new_pred)
        keys = sorted(set(old_map) | set(new_map))
        tvd = 0.5 * sum(abs(new_map.get(k, 0.0) - old_map.get(k, 0.0)) for k in keys)
        base = 0.15
        threshold = base * factor
        if tvd >= threshold:
            return True, f"tvd={tvd:.3f}"
        return False, f"tvd={tvd:.3f}"

    if question_type in {"numeric", "date", "discrete"}:
        if not isinstance(old_pred, dict) or not isinstance(new_pred, dict):
            return True, "numeric_type_changed"
        old_map = _get_numeric_percentile_map(old_pred)
        new_map = _get_numeric_percentile_map(new_pred)
        old_med = _approx_median_from_percentiles(old_map)
        new_med = _approx_median_from_percentiles(new_map)
        if old_med is None or new_med is None:
            return True, "missing_median"
        old_p10 = old_map.get(0.1)
        old_p90 = old_map.get(0.9)
        new_p10 = new_map.get(0.1)
        new_p90 = new_map.get(0.9)
        old_width = (old_p90 - old_p10) if (old_p10 is not None and old_p90 is not None) else None
        new_width = (new_p90 - new_p10) if (new_p10 is not None and new_p90 is not None) else None
        width = None
        if old_width is not None and new_width is not None:
            width = max(1e-9, 0.5 * (old_width + new_width))
        elif old_width is not None:
            width = max(1e-9, old_width)
        elif new_width is not None:
            width = max(1e-9, new_width)
        else:
            width = max(1e-9, abs(old_med), abs(new_med))

        normalized_median_shift = abs(new_med - old_med) / width
        base = 0.35
        threshold = base * factor

        if question_type == "date":
            median_shift_days = abs(new_med - old_med) / 86400
            absolute_day_threshold = 30 * factor
            if median_shift_days >= absolute_day_threshold:
                return True, f"median_shift_days={median_shift_days:.1f}"

        if normalized_median_shift >= threshold:
            return True, f"norm_median_shift={normalized_median_shift:.3f}"
        return False, f"norm_median_shift={normalized_median_shift:.3f}"

    if question_type == "conditional":
        # Conservative: treat any change in child prediction as significant by reusing the same function recursively.
        if not isinstance(old_pred, dict) or not isinstance(new_pred, dict):
            return True, "conditional_type_changed"
        old_child = old_pred.get("child")
        new_child = new_pred.get("child")
        child_type = "unknown"
        if isinstance(new_child, (int, float)):
            child_type = "binary"
        elif isinstance(new_child, dict) and "predicted_options" in new_child:
            child_type = "multiple_choice"
        elif isinstance(new_child, dict) and "declared_percentiles" in new_child:
            child_type = "numeric"
        return _is_significant_change(
            old_pred=old_child,
            new_pred=new_child,
            question_type=child_type,
            close_time_iso=close_time_iso,
            old_close_time_iso=old_close_time_iso,
        )

    return True, f"unknown_type_{question_type}"


def _matrix_send_message(message: str) -> None:
    homeserver = os.getenv("MATRIX_HOMESERVER")
    access_token = os.getenv("MATRIX_ACCESS_TOKEN")
    room_id = os.getenv("MATRIX_ROOM_ID")
    if not homeserver or not access_token or not room_id:
        return

    txn_id = uuid4().hex
    room_id_escaped = requests.utils.quote(room_id, safe="")
    url = (
        f"{homeserver.rstrip('/')}/_matrix/client/v3/rooms/"
        f"{room_id_escaped}/send/m.room.message/{txn_id}"
    )
    headers = {"Authorization": f"Bearer {access_token}"}
    payload = {"msgtype": "m.text", "body": message}
    response = requests.put(url, headers=headers, json=payload, timeout=30)
    if not response.ok:
        logger.warning(f"Matrix send failed: {response.status_code} {response.text}")


async def _run_digest(
    *,
    template_bot: SpringTemplateBot2026,
    tournaments: list[str],
    state_path: Path,
    out_dir: Path,
) -> None:
    from forecasting_tools.data_models.data_organizer import DataOrganizer

    out_dir.mkdir(parents=True, exist_ok=True)
    state_path.parent.mkdir(parents=True, exist_ok=True)

    previous_state: dict = {}
    if state_path.exists():
        try:
            previous_state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Failed to read previous state: {e}")
            previous_state = {}

    previous_questions: dict = previous_state.get("questions", {}) if isinstance(previous_state, dict) else {}
    now_iso = datetime.now(timezone.utc).isoformat()

    new_state_questions: dict[str, dict] = {}
    significant_changes: list[dict] = []
    failures: list[dict] = []

    def safe_report_attr(report: object, attr: str) -> str | None:
        try:
            return getattr(report, attr)
        except Exception:
            return None

    for tournament_id in tournaments:
        try:
            reports_or_errors = await template_bot.forecast_on_tournament(
                tournament_id, return_exceptions=True
            )
        except Exception as e:
            failures.append({"tournament": tournament_id, "error": str(e)})
            continue

        for item in reports_or_errors:
            if isinstance(item, BaseException):
                failures.append({"tournament": tournament_id, "error": repr(item)})
                continue

            report = item
            question = report.question
            key = "|".join(
                [
                    str(question.id_of_post or ""),
                    str(question.id_of_question or ""),
                    str(question.conditional_type or ""),
                    str(question.group_question_option or ""),
                ]
            )

            if isinstance(question, BinaryQuestion):
                qtype = "binary"
            elif isinstance(question, MultipleChoiceQuestion):
                qtype = "multiple_choice"
            elif isinstance(question, DateQuestion):
                qtype = "date"
            elif isinstance(question, NumericQuestion):
                qtype = "numeric"
            elif isinstance(question, ConditionalQuestion):
                qtype = "conditional"
            else:
                qtype = "unknown"

            close_time_iso = _get_close_time_iso(question)
            prediction_jsonable = _prediction_to_compact_jsonable(
                report.prediction
            )
            try:
                readable_prediction = DataOrganizer.get_readable_prediction(
                    report.prediction
                )
            except Exception as e:
                readable_prediction = f"(failed to format prediction: {e})"

            state_snapshot = {
                "generated_at": now_iso,
                "tournament": tournament_id,
                "question_text": question.question_text,
                "page_url": question.page_url,
                "id_of_post": question.id_of_post,
                "id_of_question": question.id_of_question,
                "close_time": close_time_iso,
                "question_type": qtype,
                "prediction": prediction_jsonable,
            }
            new_state_questions[key] = state_snapshot

            old_snapshot = previous_questions.get(key)
            old_pred = old_snapshot.get("prediction") if isinstance(old_snapshot, dict) else None
            old_close_time = old_snapshot.get("close_time") if isinstance(old_snapshot, dict) else None

            significant, reason = _is_significant_change(
                old_pred=old_pred,
                new_pred=prediction_jsonable,
                question_type=qtype,
                close_time_iso=close_time_iso,
                old_close_time_iso=old_close_time,
            )
            if significant:
                significant_changes.append(
                    {
                        "key": key,
                        "tournament": tournament_id,
                        "page_url": question.page_url,
                        "question_text": question.question_text,
                        "question_type": qtype,
                        "close_time": close_time_iso,
                        "reason": reason,
                        "old_prediction": old_pred,
                        "new_prediction": prediction_jsonable,
                        "readable_prediction": readable_prediction,
                        "summary": safe_report_attr(report, "summary"),
                        "research": safe_report_attr(report, "research"),
                        "forecast_rationales": safe_report_attr(
                            report, "forecast_rationales"
                        ),
                        "explanation": getattr(report, "explanation", None),
                    }
                )

    new_state = {"version": 1, "generated_at": now_iso, "questions": new_state_questions}
    state_path.write_text(json.dumps(new_state, ensure_ascii=False, indent=2), encoding="utf-8")

    digest_path = out_dir / f"digest_{now_iso[:10]}.md"
    changes_path = out_dir / "changes.md"
    failures_path = out_dir / "failures.json"

    def fmt_pred(pred: object | None) -> str:
        if pred is None:
            return "(none)"
        try:
            return json.dumps(pred, ensure_ascii=False)
        except Exception:
            return str(pred)

    lines: list[str] = []
    lines.append(f"# Metaculus digest ({now_iso})")
    lines.append("")
    lines.append("## Tournaments")
    for tid in tournaments:
        lines.append(f"- {tid}")
    lines.append("")

    lines.append(f"## Significant changes ({len(significant_changes)})")
    if not significant_changes:
        lines.append("- (none)")
    else:
        for ch in significant_changes:
            url = ch.get("page_url") or ch.get("key")
            lines.append(f"### {ch['question_text']}")
            lines.append(f"- Tournament: {ch['tournament']}")
            lines.append(f"- URL: {url}")
            if ch.get("close_time"):
                lines.append(f"- Close time (UTC): {ch['close_time']}")
            lines.append(f"- Type: {ch['question_type']}")
            lines.append(f"- Reason: {ch['reason']}")
            lines.append(f"- Old: {fmt_pred(ch['old_prediction'])}")
            lines.append(f"- New: {fmt_pred(ch['new_prediction'])}")
            lines.append("")
            lines.append("**Readable prediction**")
            lines.append(ch.get("readable_prediction") or "")
            lines.append("")
            if ch.get("summary"):
                lines.append("**Report summary**")
                lines.append(str(ch["summary"]))
                lines.append("")
            if ch.get("research"):
                lines.append("**Report research**")
                lines.append(str(ch["research"]))
                lines.append("")
            if ch.get("forecast_rationales"):
                lines.append("**Report forecast**")
                lines.append(str(ch["forecast_rationales"]))
                lines.append("")

    digest_path.write_text("\n".join(lines), encoding="utf-8")
    changes_path.write_text("\n".join(lines), encoding="utf-8")
    failures_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")

    if significant_changes:
        message_lines = [
            f"Metaculus digest: {len(significant_changes)} significant change(s) ({now_iso[:10]})",
        ]
        for ch in significant_changes[:10]:
            url = ch.get("page_url") or ch.get("key")
            message_lines.append(f"- {ch['tournament']}: {ch['question_text']} ({url})")
        if len(significant_changes) > 10:
            message_lines.append(f"... and {len(significant_changes) - 10} more")
        _matrix_send_message("\n".join(message_lines))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Suppress LiteLLM logging
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
        default="tracked_tournaments.txt",
        help="Path to a file containing tournament URLs/slugs to track (one per line).",
    )
    parser.add_argument(
        "--tournament",
        action="append",
        default=[],
        help="Tournament slug/id or URL (repeatable). Overrides/extends --tournaments-file.",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "metaculus_cup", "test_questions", "digest"] = args.mode
    assert run_mode in [
        "tournament",
        "metaculus_cup",
        "test_questions",
        "digest",
    ], "Invalid run mode"

    publish_to_metaculus = run_mode in {"tournament", "metaculus_cup", "test_questions"}
    template_bot = SpringTemplateBot2026(
        research_reports_per_question=1,
        predictions_per_research_report=5,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=publish_to_metaculus,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=(run_mode == "tournament"),
        extra_metadata_in_explanation=True,
        # llms={  # choose your model names or GeneralLlm llms here, otherwise defaults will be chosen for you
        #     "default": GeneralLlm(
        #         model="openrouter/openai/gpt-4o", # "anthropic/claude-sonnet-4-20250514", etc (see docs for litellm)
        #         temperature=0.3,
        #         timeout=40,
        #         allowed_tries=2,
        #     ),
        #     "summarizer": "openai/gpt-4o-mini",
        #     "researcher": "asknews/news-summaries",
        #     "parser": "openai/gpt-4o-mini",
        # },
    )

    client = MetaculusClient()
    if run_mode == "tournament":
        if args.tournament:
            tournaments, unsupported = _load_tournament_identifiers(
                None, args.tournament
            )
            for item in unsupported:
                logger.warning(f"Unsupported collection (not a tournament): {item}")
            if not tournaments:
                raise SystemExit("No valid tournaments provided via --tournament.")
            forecast_reports = []
            for tournament_id in tournaments:
                forecast_reports.extend(
                    asyncio.run(
                        template_bot.forecast_on_tournament(
                            tournament_id, return_exceptions=True
                        )
                    )
                )
        else:
            # You may want to change this to the specific tournament ID you want to forecast on
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
        # The Metaculus cup is a good way to test the bot's performance on regularly open questions. You can also use AXC_2025_TOURNAMENT_ID = 32564 or AI_2027_TOURNAMENT_ID = "ai-2027"
        # The Metaculus cup may not be initialized near the beginning of a season (i.e. January, May, September)
        template_bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                client.CURRENT_METACULUS_CUP_ID, return_exceptions=True
            )
        )
    elif run_mode == "test_questions":
        # Example questions are a good way to test the bot's performance on a single question
        EXAMPLE_QUESTIONS = [
            "https://www.metaculus.com/questions/578/human-extinction-by-2100/",  # Human Extinction - Binary
            "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",  # Age of Oldest Human - Numeric
            "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",  # Number of New Leading AI Labs - Multiple Choice
            "https://www.metaculus.com/c/diffusion-community/38880/how-many-us-labor-strikes-due-to-ai-in-2029/",  # Number of US Labor Strikes Due to AI in 2029 - Discrete
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
        tournaments, unsupported = _load_tournament_identifiers(
            args.tournaments_file, args.tournament
        )
        for item in unsupported:
            logger.warning(f"Unsupported collection (not a tournament): {item}")
        if not tournaments:
            raise SystemExit(
                "No tournaments configured. Add tournament URLs/slugs to tracked_tournaments.txt or pass --tournament."
            )

        template_bot.publish_reports_to_metaculus = False
        template_bot.skip_previously_forecasted_questions = False
        state_path = Path(".state") / "digest_state.json"
        out_dir = Path("reports") / "digest"
        asyncio.run(
            _run_digest(
                template_bot=template_bot,
                tournaments=tournaments,
                state_path=state_path,
                out_dir=out_dir,
            )
        )
        forecast_reports = []
    template_bot.log_report_summary(forecast_reports)
