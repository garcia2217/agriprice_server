"""
Data validation service for handling comprehensive data validation.

This service orchestrates all validation steps and provides structured
results with clear error messages and context.
"""

import time
import io
from typing import Dict, Any, List, Tuple
from pathlib import Path
import pandas as pd

from src.preprocessing import ConsolidationConfig, DataConsolidator
from src.utils.validators import (
    validate_file_format, validate_geographic_scope, validate_data_structure,
    validate_commodities, extract_available_options, validate_zip_structure
)
from ..exceptions import (
    FileFormatError, DataStructureError, GeographicScopeError,
    CommodityValidationError, ConsolidationError
)
from ..constants import (
    MAX_FILE_SIZE_BYTES, MIN_REQUIRED_RECORDS, MIN_DATE_RANGE_DAYS,
    MIN_COMMODITIES_FOR_RADAR, ErrorMessages
)


class ValidationResult:
    """Container for validation results with structured data."""
    
    def __init__(
        self, 
        is_valid: bool, 
        data: pd.DataFrame = None, 
        errors: List[str] = None,
        warnings: List[str] = None,
        metrics: Dict[str, Any] = None
    ):
        self.is_valid = is_valid
        self.data = data
        self.errors = errors or []
        self.warnings = warnings or []
        self.metrics = metrics or {}


class DataValidationService:
    """Service for comprehensive data validation."""
    
    def __init__(self):
        """Initialize validation service."""
        self.config = ConsolidationConfig(input_type="zip")
    
    def validate_file_upload(self, file) -> None:
        """
        Validate uploaded file format and size.
        
        Args:
            file: Uploaded file object
            
        Raises:
            FileFormatError: If file validation fails
        """
        if not file:
            raise FileFormatError(ErrorMessages.NO_FILE_PROVIDED)
        
        # Check file size
        if file.size > MAX_FILE_SIZE_BYTES:
            size_mb = file.size / (1024 * 1024)
            raise FileFormatError(
                ErrorMessages.FILE_TOO_LARGE.format(
                    size_mb=size_mb,
                    max_mb=MAX_FILE_SIZE_BYTES / (1024 * 1024)
                ),
                details={
                    "file_size_bytes": file.size,
                    "file_size_mb": size_mb,
                    "max_size_bytes": MAX_FILE_SIZE_BYTES,
                    "max_size_mb": MAX_FILE_SIZE_BYTES / (1024 * 1024)
                }
            )
        
        # Validate file format using existing validator
        is_valid_format, format_errors = validate_file_format(file)
        if not is_valid_format:
            raise FileFormatError(
                format_errors[0] if format_errors else ErrorMessages.INVALID_ZIP_FORMAT,
                details={"format_errors": format_errors}
            )
    
    def validate_data_content(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Validate data structure, geography, and commodities.
        
        Args:
            df: Consolidated dataframe to validate
            
        Returns:
            Tuple of (errors, warnings)
            
        Raises:
            DataStructureError: If data structure validation fails
            GeographicScopeError: If geographic scope validation fails
            CommodityValidationError: If commodity validation fails
        """
        errors = []
        warnings = []
        
        # 1. Data structure validation
        is_valid_structure, structure_errors = validate_data_structure(df)
        if not is_valid_structure:
            # Enhance error messages with context
            enhanced_errors = self._enhance_structure_errors(structure_errors, df)
            errors.extend(enhanced_errors)
        
        # 2. Geographic scope validation
        valid_cities = self.config.valid_cities
        is_valid_geo, geo_errors = validate_geographic_scope(df, valid_cities)
        if not is_valid_geo:
            # Enhance error messages with valid cities list
            enhanced_errors = self._enhance_geographic_errors(geo_errors, valid_cities)
            errors.extend(enhanced_errors)
        
        # 3. Commodity validation
        supported_commodities = self.config.commodities
        is_valid_commodities, commodity_errors = validate_commodities(df, supported_commodities)
        if not is_valid_commodities:
            # Enhance error messages with supported commodities
            enhanced_errors = self._enhance_commodity_errors(commodity_errors, supported_commodities)
            errors.extend(enhanced_errors)
        
        return errors, warnings
    
    def run_all_validations(self, file) -> ValidationResult:
        """
        Run complete validation pipeline.
        
        Args:
            file: Uploaded file object
            
        Returns:
            ValidationResult with validation status and data
        """
        # Step 1: Validate file upload
        self.validate_file_upload(file)
        
        # Step 2: Validate ZIP structure (PROVINSI → KOTA → YYYY.xlsx)
        is_valid_structure, structure_errors = validate_zip_structure(file)
        if not is_valid_structure:
            raise FileFormatError(
                structure_errors[0] if structure_errors else "Invalid ZIP structure",
                details={"structure_errors": structure_errors}
            )
        
        # Step 3: Process ZIP file using consolidation pipeline
        try:
            consolidator = DataConsolidator(self.config)
            
            # Reset file position and read bytes
            file.seek(0)
            zip_bytes = file.read()
            zip_stream = io.BytesIO(zip_bytes)
            zip_stream.seek(0)
            
            results = consolidator.process_zip_stream(zip_stream=zip_stream)
            
            if not results.get("success", False):
                raise ConsolidationError(
                    ErrorMessages.CONSOLIDATION_FAILED.format(
                        error=results.get('error', 'Unknown error')
                    ),
                    details={"consolidation_error": results.get('error')}
                )
            
            consolidated_df = results.get("consolidated_df")
            if consolidated_df is None:
                raise ConsolidationError(
                    ErrorMessages.NO_DATA_CONSOLIDATED,
                    details={"consolidation_results": results}
                )
            
        except Exception as e:
            if isinstance(e, ConsolidationError):
                raise
            raise ConsolidationError(
                ErrorMessages.CONSOLIDATION_FAILED.format(error=str(e)),
                details={"original_error": str(e)}
            )
        
        # Step 4: Validate data content
        errors, warnings = self.validate_data_content(consolidated_df)
        
        # Step 5: Calculate metrics
        metrics = self._calculate_validation_metrics(consolidated_df)
        
        # Return validation result
        is_valid = len(errors) == 0
        return ValidationResult(
            is_valid=is_valid,
            data=consolidated_df if is_valid else None,
            errors=errors,
            warnings=warnings,
            metrics=metrics
        )
    
    def _enhance_structure_errors(self, errors: List[str], df: pd.DataFrame) -> List[str]:
        """Enhance data structure error messages with context."""
        enhanced_errors = []
        
        for error in errors:
            if "Missing required columns" in error:
                # Add context about what columns are available
                available_columns = list(df.columns)
                enhanced_errors.append(
                    f"{error}. Available columns: {available_columns}"
                )
            elif "Insufficient data" in error:
                # Add context about data size
                enhanced_errors.append(
                    f"{error}. Current data has {len(df)} records."
                )
            else:
                enhanced_errors.append(error)
        
        return enhanced_errors
    
    def _enhance_geographic_errors(self, errors: List[str], valid_cities: List[str]) -> List[str]:
        """Enhance geographic error messages with valid cities."""
        enhanced_errors = []
        
        for error in errors:
            if "Invalid cities found" in error:
                # Add list of valid cities for reference
                enhanced_errors.append(
                    f"{error}. Supported cities: {sorted(valid_cities)[:10]}{'...' if len(valid_cities) > 10 else ''}"
                )
            else:
                enhanced_errors.append(error)
        
        return enhanced_errors
    
    def _enhance_commodity_errors(self, errors: List[str], supported_commodities: List[str]) -> List[str]:
        """Enhance commodity error messages with supported commodities."""
        enhanced_errors = []
        
        for error in errors:
            if "Unsupported commodities found" in error:
                # Add list of supported commodities
                enhanced_errors.append(
                    f"{error}. Supported commodities: {sorted(supported_commodities)}"
                )
            elif "Insufficient commodities" in error:
                # Add context about minimum requirements
                enhanced_errors.append(
                    f"{error}. Minimum {MIN_COMMODITIES_FOR_RADAR} commodities required for radar chart."
                )
            else:
                enhanced_errors.append(error)
        
        return enhanced_errors
    
    def _calculate_validation_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate validation metrics for logging and monitoring."""
        return {
            "total_records": len(df),
            "total_cities": df['City'].nunique() if 'City' in df.columns else 0,
            "total_commodities": df['Commodity'].nunique() if 'Commodity' in df.columns else 0,
            "date_range_days": self._calculate_date_range(df),
            "validation_timestamp": time.time()
        }
    
    def _calculate_date_range(self, df: pd.DataFrame) -> int:
        """Calculate date range in days."""
        if 'Date' not in df.columns:
            return 0
        
        try:
            df_copy = df.copy()
            df_copy['Date'] = pd.to_datetime(df_copy['Date'])
            return (df_copy['Date'].max() - df_copy['Date'].min()).days
        except:
            return 0
