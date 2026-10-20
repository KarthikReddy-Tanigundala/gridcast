"""One logging format for every ``python -m gridcast.<module>`` entrypoint."""

from __future__ import annotations

import logging

FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging and quieten chatty third-party loggers."""
    logging.basicConfig(level=level, format=FORMAT, datefmt="%H:%M:%S", force=True)
    for noisy in ("urllib3", "numba", "lightgbm", "matplotlib", "fsspec", "cmdstanpy"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
