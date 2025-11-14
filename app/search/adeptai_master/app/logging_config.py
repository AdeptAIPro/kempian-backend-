import logging
import os
from logging.config import dictConfig


def configure_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "standard",
                    "level": level,
                }
            },
            "root": {"handlers": ["console"], "level": level},
        }
    )
    logging.getLogger(__name__).debug("Logging configured at %s", level)


