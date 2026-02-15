"""
Logging configuration for the MLForecast application.

This module provides structured logging setup for both development
and production environments.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Literal

LOG_FORMAT_DEV = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_FORMAT_PROD = (
    '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
    '"logger": "%(name)s", "message": "%(message)s"}'
)


def setup_logging(
    level: str | None = None,
    environment: Literal["dev", "test", "prod"] = "dev",
) -> None:
    """
    Configure application logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR). Defaults to INFO.
        environment: Environment name to determine format (dev=readable, prod=JSON).
    """
    log_level = level or os.getenv("LOG_LEVEL", "INFO")
    log_format = LOG_FORMAT_PROD if environment == "prod" else LOG_FORMAT_DEV

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )

    # Reduce noise from third-party libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("lightgbm").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    return logging.getLogger(name)

