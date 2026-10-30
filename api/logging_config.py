"""Application logging configuration, completed in Task 3."""

import logging


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="[{levelname}] {name}: {message}", style="{")

    root = logging.getLogger()
    root.setLevel(level)

    """Configure application logging at the requested level, defaulting to INFO.
    Module loggers inherit this configuration through normal propagation. The
    logging exercise runs entry operations at INFO and DEBUG to compare output.
    Never log journal text, settings objects, or credentials.
    """
