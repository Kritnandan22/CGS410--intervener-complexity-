"""Structured logging setup with file + console handlers."""

from __future__ import annotations

import logging

import os

from datetime import datetime

def setup_logging(

    log_dir: str,

    language: str = "global",

    level: str = "INFO",

) -> logging.Logger:

    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_file = os.path.join(log_dir, f"{language}_{timestamp}.log")

    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

    logging.basicConfig(

        level=getattr(logging, level.upper(), logging.INFO),

        format=fmt,

        handlers=[

            logging.FileHandler(log_file, encoding="utf-8"),

            logging.StreamHandler(),

        ],

        force=True,

    )

    logger = logging.getLogger(language)

    logger.info("Logging initialized: %s", log_file)

    return logger

