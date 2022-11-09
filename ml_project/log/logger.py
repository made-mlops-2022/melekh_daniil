import logging
import sys

LOG_FORMAT_FILE = '[%(asctime)s] - %(levelname)s - %(name)s - %(message)s'


def get_logger(name: str) -> logging.Logger:
    """
    Get logger

    :param name: name for logger

    :return: logger
    """

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter_file = logging.Formatter(LOG_FORMAT_FILE)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter_file)

    logger.addHandler(handler)

    return logger
