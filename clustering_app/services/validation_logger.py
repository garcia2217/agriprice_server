"""
Validation logger service for structured logging.

This service provides comprehensive logging for validation processes
with structured data for monitoring and debugging.
"""

import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime


class ValidationLogger:
    """Service for structured validation logging."""
    
    def __init__(self):
        """Initialize validation logger."""
        self.logger = logging.getLogger('validation')
        
        # Configure logger if not already configured
        if not self.logger.handlers:
            self._setup_logger()
    
    def _setup_logger(self):
        """Setup logger configuration."""
        self.logger.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(console_handler)
    
    def log_validation_start(self, file_name: str, file_size: int):
        """
        Log validation process start.
        
        Args:
            file_name: Name of uploaded file
            file_size: Size of file in bytes
        """
        self.logger.info(
            "Validation started",
            extra={
                "event": "validation_start",
                "file_name": file_name,
                "file_size_bytes": file_size,
                "file_size_mb": round(file_size / (1024 * 1024), 2),
                "timestamp": datetime.now().isoformat()
            }
        )
    
    def log_validation_step(self, step_name: str, status: str, duration: float = None):
        """
        Log validation step completion.
        
        Args:
            step_name: Name of validation step
            status: Status of step (success, error, warning)
            duration: Duration of step in seconds
        """
        log_data = {
            "event": "validation_step",
            "step": step_name,
            "status": status,
            "timestamp": datetime.now().isoformat()
        }
        
        if duration is not None:
            log_data["duration_seconds"] = duration
        
        if status == "success":
            self.logger.info(f"Validation step '{step_name}' completed successfully", extra=log_data)
        elif status == "error":
            self.logger.error(f"Validation step '{step_name}' failed", extra=log_data)
        else:
            self.logger.warning(f"Validation step '{step_name}' completed with warnings", extra=log_data)
    
    def log_validation_error(self, step: str, error: str, details: Dict[str, Any] = None):
        """
        Log validation error.
        
        Args:
            step: Validation step where error occurred
            error: Error message
            details: Additional error details
        """
        self.logger.error(
            f"Validation error in step '{step}': {error}",
            extra={
                "event": "validation_error",
                "step": step,
                "error": error,
                "details": details or {},
                "timestamp": datetime.now().isoformat()
            }
        )
    
    def log_validation_success(self, validation_id: str, metrics: Dict[str, Any] = None):
        """
        Log successful validation completion.
        
        Args:
            validation_id: Unique validation identifier
            metrics: Validation metrics
        """
        self.logger.info(
            "Validation completed successfully",
            extra={
                "event": "validation_success",
                "validation_id": validation_id,
                "metrics": metrics or {},
                "timestamp": datetime.now().isoformat()
            }
        )
    
    def log_file_processing(self, operation: str, file_path: str, success: bool, error: str = None):
        """
        Log file processing operations.
        
        Args:
            operation: Type of file operation (save, load, delete)
            file_path: Path to file being processed
            success: Whether operation was successful
            error: Error message if operation failed
        """
        log_data = {
            "event": "file_processing",
            "operation": operation,
            "file_path": str(file_path),
            "success": success,
            "timestamp": datetime.now().isoformat()
        }
        
        if error:
            log_data["error"] = error
        
        if success:
            self.logger.info(f"File operation '{operation}' completed", extra=log_data)
        else:
            self.logger.error(f"File operation '{operation}' failed: {error}", extra=log_data)
    
    def log_cleanup_operation(self, files_cleaned: int, age_threshold_hours: int):
        """
        Log cleanup operations.
        
        Args:
            files_cleaned: Number of files cleaned up
            age_threshold_hours: Age threshold used for cleanup
        """
        self.logger.info(
            f"Cleanup completed: {files_cleaned} files removed",
            extra={
                "event": "cleanup",
                "files_cleaned": files_cleaned,
                "age_threshold_hours": age_threshold_hours,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    def log_performance_metrics(self, operation: str, duration: float, metrics: Dict[str, Any] = None):
        """
        Log performance metrics.
        
        Args:
            operation: Name of operation
            duration: Duration in seconds
            metrics: Additional performance metrics
        """
        self.logger.info(
            f"Performance metrics for '{operation}': {duration:.2f}s",
            extra={
                "event": "performance",
                "operation": operation,
                "duration_seconds": duration,
                "metrics": metrics or {},
                "timestamp": datetime.now().isoformat()
            }
        )
