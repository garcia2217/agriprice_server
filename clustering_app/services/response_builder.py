"""
Response builder service for standardized API responses.

This service provides consistent response formatting for validation
endpoints with proper error handling and structured data.
"""

from typing import Dict, Any, List, Optional
from django.http import JsonResponse

from ..exceptions import ValidationError


class ValidationResponseBuilder:
    """Service for building standardized validation responses."""
    
    def build_success_response(
        self, 
        validation_id: str, 
        available_data: Dict[str, Any], 
        warnings: List[str] = None
    ) -> JsonResponse:
        """
        Build success response for validation.
        
        Args:
            validation_id: Unique validation identifier
            available_data: Extracted data options
            warnings: List of warnings
            
        Returns:
            JsonResponse with success data
        """
        response_data = {
            "validation_id": validation_id,
            "valid": True,
            "errors": [],
            "warnings": warnings or [],
            "available_data": available_data
        }
        
        return JsonResponse(response_data, status=200)
    
    def build_error_response(
        self, 
        errors: List[str], 
        status_code: int = 400,
        error_type: str = "VALIDATION_ERROR",
        details: Dict[str, Any] = None
    ) -> JsonResponse:
        """
        Build error response for validation failures.
        
        Args:
            errors: List of error messages
            status_code: HTTP status code
            error_type: Type of error
            details: Additional error details
            
        Returns:
            JsonResponse with error data
        """
        response_data = {
            "valid": False,
            "error_type": error_type,
            "message": errors[0] if errors else "Validation failed",
            "errors": errors,
            "details": details or {}
        }
        
        return JsonResponse(response_data, status=status_code)
    
    def build_validation_error_response(self, validation_error: ValidationError) -> JsonResponse:
        """
        Build error response from ValidationError exception.
        
        Args:
            validation_error: ValidationError exception
            
        Returns:
            JsonResponse with structured error data
        """
        response_data = {
            "valid": False,
            "error_type": validation_error.error_type,
            "message": validation_error.user_message,
            "errors": [validation_error.user_message],
            "details": validation_error.details
        }
        
        return JsonResponse(response_data, status=validation_error.http_status)
    
    def build_system_error_response(self, error_message: str, original_error: str = None) -> JsonResponse:
        """
        Build system error response for unexpected errors.
        
        Args:
            error_message: User-friendly error message
            original_error: Original error details for debugging
            
        Returns:
            JsonResponse with system error data
        """
        response_data = {
            "valid": False,
            "error_type": "SYSTEM_ERROR",
            "message": error_message,
            "errors": [error_message],
            "details": {
                "error": original_error,
                "timestamp": self._get_timestamp()
            }
        }
        
        return JsonResponse(response_data, status=500)
    
    def build_method_not_allowed_response(self) -> JsonResponse:
        """
        Build method not allowed response.
        
        Returns:
            JsonResponse for method not allowed
        """
        response_data = {
            "valid": False,
            "error_type": "METHOD_NOT_ALLOWED",
            "message": "Use POST method",
            "errors": ["Use POST method"]
        }
        
        return JsonResponse(response_data, status=405)
    
    def build_validation_result_response(self, validation_result) -> JsonResponse:
        """
        Build response from ValidationResult object.
        
        Args:
            validation_result: ValidationResult object
            
        Returns:
            JsonResponse based on validation result
        """
        if validation_result.is_valid:
            # For successful validation, we need available_data
            # This would typically be extracted from the validated data
            available_data = self._extract_available_data(validation_result.data)
            return self.build_success_response(
                validation_id="",  # This should be provided by caller
                available_data=available_data,
                warnings=validation_result.warnings
            )
        else:
            return self.build_error_response(
                errors=validation_result.errors,
                status_code=400
            )
    
    def _extract_available_data(self, df) -> Dict[str, Any]:
        """
        Extract available data options from validated dataframe.
        
        Args:
            df: Validated dataframe
            
        Returns:
            Dictionary with available data options
        """
        # This is a placeholder - in practice, this would use
        # the existing extract_available_options function
        return {
            "commodities": [],
            "cities": [],
            "years": [],
            "provinces": {},
            "date_range": {"start": None, "end": None},
            "total_records": len(df) if df is not None else 0
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.now().isoformat()
