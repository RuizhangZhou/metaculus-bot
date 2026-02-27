"""
Compatibility shim for the Spring 2026 template bot.

The implementation has been refactored into smaller modules under `bot/`,
but `main.py`, tests, and existing integrations still import from
`template_bot_2026`.
"""

from bot.spring_template_bot_2026 import SpringTemplateBot2026, _extract_probability_percent

__all__ = [
    "SpringTemplateBot2026",
    "_extract_probability_percent",
]

