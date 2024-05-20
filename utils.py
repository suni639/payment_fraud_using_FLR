import logging

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)
