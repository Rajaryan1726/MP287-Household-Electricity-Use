import json
import logging
import sys


class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {"level": record.levelname, "message": record.getMessage()}
        payload.update(getattr(record, "extra_data", {}))
        return json.dumps(payload)


def get_logger():
    logger = logging.getLogger("mp287")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger