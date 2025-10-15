"""
Logging utilities for the food price clustering project.
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logging(
    enable_file_logging: bool = False,
    log_level: str = "INFO",
    log_dir: Path = Path("logs"),
    module_name: str = "food_price_clustering"
) -> logging.Logger:
    """
    Setup logging with optional file output.
    
    Args:
        enable_file_logging: Whether to save logs to file
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory to save log files
        module_name: Name of the logger module
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(module_name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Console handler (always enabled)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if enable_file_logging:
        # Create logs directory
        log_dir.mkdir(exist_ok=True)
        
        # Generate timestamped log filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = log_dir / f"{module_name}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # Log everything to file
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"File logging enabled - Log file: {log_filename}")
    else:
        logger.info("Console logging only (file logging disabled)")
    
    return logger

