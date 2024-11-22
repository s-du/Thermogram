from loguru import logger
from utils.config import config

# Configure logger
logger.remove()  # Remove default handler
logger.add(
    config.LOG_FILE,
    rotation="10 MB",
    retention="1 month",
    level=config.LOG_LEVEL,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{function}:{line} | {message}"
)

# Create convenience functions
def info(msg: str) -> None:
    """Log info message"""
    logger.info(msg)

def error(msg: str) -> None:
    """Log error message"""
    logger.error(msg)

def debug(msg: str) -> None:
    """Log debug message"""
    logger.debug(msg)

def warning(msg: str) -> None:
    """Log warning message"""
    logger.warning(msg)