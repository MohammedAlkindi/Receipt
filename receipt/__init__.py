"""receipt — intelligent personal finance analysis engine."""

import logging
import os

__version__ = "0.1.0"
__author__ = "Mohammed Alkindi"


def configure_logging() -> None:
    level = os.getenv("RECEIPT_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, level, logging.INFO))
