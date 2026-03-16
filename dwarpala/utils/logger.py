"""
Dwarpala logging utilities using loguru.
"""

import sys
from loguru import logger


def get_logger(name: str = "dwarpala") -> logger.__class__:
    """
    Configure and return a loguru logger instance.

    Args:
        name: Module name for log context.

    Returns:
        Configured loguru logger.
    """
    # Remove default handler
    logger.remove()

    # Console handler with rich formatting
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{extra[module]}</cyan> | "
            "<level>{message}</level>"
        ),
        level="INFO",
        colorize=True,
    )

    # Bind the module name
    return logger.bind(module=name)
