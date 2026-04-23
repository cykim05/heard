"""Project-wide logger. Thin wrapper so scripts all emit identical format."""
from __future__ import annotations

import logging
import sys
from pathlib import Path


_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, *, level: int = logging.INFO, file: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    logger.addHandler(handler)

    if file is not None:
        file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(file, encoding="utf-8")
        fh.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
        logger.addHandler(fh)

    logger.propagate = False
    return logger
