from __future__ import annotations

import logging


LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


def setup_logging(level: str = "INFO") -> None:
    root_logger = logging.getLogger()
    normalized_level = getattr(logging, level.upper(), logging.INFO)

    if not root_logger.handlers:
        logging.basicConfig(level=normalized_level, format=LOG_FORMAT)
    else:
        root_logger.setLevel(normalized_level)
        for handler in root_logger.handlers:
            handler.setLevel(normalized_level)
            handler.setFormatter(logging.Formatter(LOG_FORMAT))


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
