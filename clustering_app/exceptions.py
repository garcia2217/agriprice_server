"""
Custom exceptions for validation and data processing.

These exceptions provide structured error handling with clear messages
and context for debugging and user feedback.
"""

from typing import Dict, Any, Optional


class ValidationError(Exception):
    """Base exception for all validation errors."""
    
    def __init__(
        self, 
        error_type: str, 
        user_message: str, 
        details: Optional[Dict[str, Any]] = None,
        http_status: int = 400
    ):
        self.error_type = error_type
        self.user_message = user_message
        self.details = details or {}
        self.http_status = http_status
        super().__init__(user_message)


class FileFormatError(ValidationError):
    """Raised when file format validation fails."""
    
    def __init__(self, user_message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            error_type="FILE_FORMAT_ERROR",
            user_message=user_message,
            details=details,
            http_status=400
        )


class DataStructureError(ValidationError):
    """Raised when data structure validation fails."""
    
    def __init__(self, user_message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            error_type="DATA_STRUCTURE_ERROR", 
            user_message=user_message,
            details=details,
            http_status=400
        )


class GeographicScopeError(ValidationError):
    """Raised when geographic scope validation fails."""
    
    def __init__(self, user_message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            error_type="GEOGRAPHIC_SCOPE_ERROR",
            user_message=user_message,
            details=details,
            http_status=400
        )


class CommodityValidationError(ValidationError):
    """Raised when commodity validation fails."""
    
    def __init__(self, user_message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            error_type="COMMODITY_VALIDATION_ERROR",
            user_message=user_message,
            details=details,
            http_status=400
        )


class ConsolidationError(ValidationError):
    """Raised when data consolidation fails."""
    
    def __init__(self, user_message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            error_type="CONSOLIDATION_ERROR",
            user_message=user_message,
            details=details,
            http_status=400
        )


class SystemError(ValidationError):
    """Raised when unexpected system errors occur."""
    
    def __init__(self, user_message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            error_type="SYSTEM_ERROR",
            user_message=user_message,
            details=details,
            http_status=500
        )
