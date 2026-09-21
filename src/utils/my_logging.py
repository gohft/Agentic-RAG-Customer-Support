import logging
import logging.config
import os
import sys

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {"format": "%(asctime)s | %(name)s | %(levelname)s | %(message)s"}
    },
    "handlers": {
        "dataset_generator": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "logs/eval.log",
            "formatter": "standard",
            "maxBytes": 5_000_000,  # 5 MB per file
            "backupCount": 3,  # keep 3 old files
            "encoding": "utf-8", 
        },
        "graph": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "logs/graph.log",
            "formatter": "standard",
            "maxBytes": 5_000_000,
            "backupCount": 3,
            "encoding": "utf-8", 
        },
        "db": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "logs/db.log",
            "formatter": "standard",
            "maxBytes": 5_000_000,
            "backupCount": 3,
            "encoding": "utf-8", 
        },
        "console": {
            "class": "logging.StreamHandler",  # print to console
            "formatter": "standard",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "dataset_generator": {
            "handlers": ["dataset_generator", "console"],
            "level": "INFO",
            "propagate": False,
        },
        "graph": {
            "handlers": ["graph", "console"],
            "level": "INFO",
            "propagate": False,
        },
        "db": {
            "handlers": ["db", "console"],
            "level": "INFO",
            "propagate": False,
        },
    },
}


def setup_logging():
    """Helper function to configure the logging config."""
    os.makedirs("logs", exist_ok=True)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.config.dictConfig(LOGGING_CONFIG)
