"""
Compatibility shim for the Spring 2026 template bot.

The implementation lives under `bot/`, but `main.py`, tests, and existing
integrations still import from `template_bot_2026`.
"""

from bot.spring_template_bot_2026 import (
    SpringTemplateBot2026 as _SpringTemplateBot2026,
    _extract_probability_percent as _impl_extract_probability_percent,
)


class SpringTemplateBot2026(_SpringTemplateBot2026):
    pass


def _extract_probability_percent(text: str) -> float | None:
    return _impl_extract_probability_percent(text)


__all__ = ["SpringTemplateBot2026", "_extract_probability_percent"]


if __name__ == "__main__":
    import runpy

    runpy.run_module("bot.spring_template_bot_2026", run_name="__main__")
