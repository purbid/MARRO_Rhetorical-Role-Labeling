import logging
from datetime import datetime





def fetch_logger(logger_name, ):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    return logger