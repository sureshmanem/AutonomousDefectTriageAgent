"""
Centralized logging configuration for the Defect Triage Agent.

This module provides consistent logging setup across all components.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from datetime import datetime


def setup_logging(
    log_level: str = "INFO",
    log_to_file: bool = True,
    log_dir: str = "./data/logs",
    log_format: str = "detailed"
) -> None:
    """
    Configure logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to file in addition to console
        log_dir: Directory for log files
        log_format: Format style - 'simple', 'detailed', or 'json'
    """
    # Create log directory if it doesn't exist
    if log_to_file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
    
    # Define log formats
    formats = {
        "simple": "%(levelname)s - %(name)s - %(message)s",
        "detailed": "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
        "json": '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "file": "%(filename)s", "line": %(lineno)d, "message": "%(message)s"}'
    }
    
    log_format_string = formats.get(log_format, formats["detailed"])
    
    # Create formatters
    formatter = logging.Formatter(
        log_format_string,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (rotating)
    if log_to_file:
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = Path(log_dir) / f"defect_triage_{timestamp}.log"
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        logging.info(f"Logging to file: {log_file}")
    
    # Set specific loggers
    # Reduce verbosity of third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("faiss").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.INFO)
    
    logging.info(f"Logging configured - Level: {log_level}, Format: {log_format}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Name of the logger (usually __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Example usage patterns for different scenarios
LOGGING_EXAMPLES = """
# Usage Examples:

1. Basic Setup (in main script):
   ```python
   from logging_config import setup_logging
   
   setup_logging(log_level="INFO", log_to_file=True)
   ```

2. Module-specific Logger:
   ```python
   import logging
   logger = logging.getLogger(__name__)
   
   logger.debug("Detailed debug information")
   logger.info("General information")
   logger.warning("Warning message")
   logger.error("Error occurred")
   logger.critical("Critical issue")
   ```

3. Logging with Variables:
   ```python
   logger.info(f"Processing file: {filename}, size: {size} bytes")
   logger.error(f"Failed to connect to {endpoint}: {error}")
   ```

4. Exception Logging:
   ```python
   try:
       risky_operation()
   except Exception as e:
       logger.exception("Operation failed with exception")
       # This automatically includes stack trace
   ```

5. Structured Logging:
   ```python
   logger.info("Request received", extra={
       "user_id": user_id,
       "endpoint": endpoint,
       "method": method
   })
   ```

6. Performance Logging:
   ```python
   import time
   start_time = time.time()
   
   # ... operation ...
   
   duration = time.time() - start_time
   logger.info(f"Operation completed in {duration:.2f}s")
   ```

7. Conditional Logging:
   ```python
   if logger.isEnabledFor(logging.DEBUG):
       # Only compute expensive debug info if DEBUG is enabled
       debug_info = expensive_operation()
       logger.debug(f"Debug info: {debug_info}")
   ```
"""


if __name__ == "__main__":
    # Demo different logging levels
    setup_logging(log_level="DEBUG", log_to_file=True, log_format="detailed")
    
    logger = get_logger(__name__)
    
    logger.debug("This is a DEBUG message - detailed diagnostic information")
    logger.info("This is an INFO message - general information")
    logger.warning("This is a WARNING message - something unexpected happened")
    logger.error("This is an ERROR message - something failed")
    
    # Demo exception logging
    try:
        1 / 0
    except Exception:
        logger.exception("This is an EXCEPTION log with traceback")
    
    print("\n" + "="*80)
    print("Logging Examples:")
    print("="*80)
    print(LOGGING_EXAMPLES)
