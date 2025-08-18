import logging
from logging.handlers import RotatingFileHandler


def setup_logger(name: str, log_file: str, level=logging.INFO):
    """
    Sets up and configures a logger with both file and console handlers. The logger
    includes logging format, log rotation for the file handler, and allows specifying
    the logging level.

    Args:
        name (str): The name of the logger.
        log_file (str): The file path for the log output. The file handler will
            log messages to this file.
        level (int | str): The logging level (e.g., logging.INFO, logging.DEBUG).
            Controls the threshold for what log messages will be handled.

    Returns:
        logging.Logger: A configured logger instance with the specified name,
        file handler, and console handler.
    """
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=5*1024*1024,  # 5MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger