import logging

def get_logger(name: str) -> logging.Logger:
    """
    Creates and configures a logger with the specified name.

    Parameters:
        name (str): The name of the logger. Convention is to use __name__.

    Returns:
        logging.Logger: The configured logger object.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("[%(levelname)s] - %(asctime)s - %(message)s")
    )
    logger.addHandler(handler)
    return logger