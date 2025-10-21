"""
Constants and configuration for validation and data processing.

This module contains all constants used throughout the validation
pipeline to ensure consistency and easy maintenance.
"""

# File size limits
MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# Validation data retention
MAX_VALIDATION_AGE_HOURS = 1

# Data quality requirements
MIN_REQUIRED_RECORDS = 100
MIN_DATE_RANGE_DAYS = 30
MIN_COMMODITIES_FOR_RADAR = 3

# Required columns for data structure validation
REQUIRED_COLUMNS = ['City', 'Commodity', 'Date', 'Price']

# Supported year range
MIN_YEAR = 2020
MAX_YEAR = 2024

# Error message templates
class ErrorMessages:
    """Templates for error messages with named placeholders."""
    
    # File format errors
    NO_FILE_PROVIDED = "No file provided in request"
    FILE_TOO_LARGE = "File size {size_mb:.1f}MB exceeds maximum {max_mb}MB"
    INVALID_ZIP_FORMAT = "Invalid ZIP file format. Please ensure the file is not corrupted."
    EMPTY_ZIP_FILE = "ZIP file is empty"
    ZIP_READ_ERROR = "Error reading ZIP file: {error}"
    
    # Data structure errors
    MISSING_COLUMNS = "Missing required columns: {missing_columns}"
    PRICE_NOT_NUMERIC = "Price column must be numeric"
    NEGATIVE_PRICES = "Price values cannot be negative"
    INSUFFICIENT_RECORDS = "Insufficient data: only {count} records found (minimum {min_count})"
    INSUFFICIENT_DATE_RANGE = "Insufficient date range: {days} days (minimum {min_days} days)"
    INVALID_DATE_FORMAT = "Invalid date format: {error}"
    
    # Geographic scope errors
    CITY_COLUMN_MISSING = "City column not found in data"
    INVALID_CITIES = "Invalid cities found: {invalid_cities} - not in supported scope"
    
    # Commodity errors
    COMMODITY_COLUMN_MISSING = "Commodity column not found"
    UNSUPPORTED_COMMODITIES = "Unsupported commodities found: {unsupported}"
    INSUFFICIENT_COMMODITIES = "Insufficient commodities: {count} found (minimum {min_count} for radar chart)"
    
    # Consolidation errors
    CONSOLIDATION_FAILED = "Data consolidation failed: {error}"
    NO_DATA_CONSOLIDATED = "Failed to consolidate data from ZIP file"
    
    # System errors
    UNEXPECTED_ERROR = "An unexpected error occurred during validation"
    VALIDATION_DATA_NOT_FOUND = "Validation data not found"
