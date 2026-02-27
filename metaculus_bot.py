"""
Public entrypoints for this repository's Metaculus bot.

Implementation details live under `bot/`.
"""

from bot.metaculus_bot import MetaculusBot, _extract_probability_percent

__all__ = ["MetaculusBot", "_extract_probability_percent"]


if __name__ == "__main__":
    import runpy

    runpy.run_module("bot.metaculus_bot", run_name="__main__")
