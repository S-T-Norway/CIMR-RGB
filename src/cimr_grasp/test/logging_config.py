# logging_config.py
import logging
import json

def configure_logging(config_file='logger_config.json'):
    # Read logging configuration from a JSON file
    with open(config_file, 'r') as file:
        config = json.load(file)

    # Check if logging is enabled
    if not config.get("logging_enabled", True):
        return None  # No logging if logging is disabled

    # Configure logging level
    log_level = getattr(logging, config.get("log_level", "INFO").upper(), logging.INFO)

    # Set up logging
    logger = logging.getLogger('my_library')
    logger.setLevel(log_level)

    if config.get("log_to_file", False):
        handler = logging.FileHandler(config.get("log_file", "app.log"))
    else:
        handler = logging.StreamHandler()

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger

