"""
Structured logging setup.

Creates a logger that writes to both console and a log file.
Every module uses this instead of raw print() for consistent output.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(
    name: str,
    log_dir: str = "outputs/logs",
    level: int = logging.INFO,
    console: bool = True,
) -> logging.Logger:
    """
    Create a named logger with console and file handlers.

    Args:
        name:    Logger name (usually module name like 'data.openi').
        log_dir: Directory for log files.
        level:   Logging level.
        console: Whether to also log to stdout.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(log_path / f"{name}_{timestamp}.log", encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler
    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger
