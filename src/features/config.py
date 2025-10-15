"""
Configuration for Feature Engineering Pipeline

This module defines the configuration class for feature engineering operations,
including feature extraction parameters, scaling options, and export settings.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field, field_validator


class FeatureEngineeringConfig(BaseModel):
    """
    Configuration class for feature engineering pipeline.
    
    Uses Pydantic for validation and type checking.
    """
    
    # Logging configuration
    enable_file_logging: bool = Field(
        default=False,
        description="Whether to save logs to file (True) or console only (False)"
    )
    
    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR"
    )
    
    log_dir: Path = Field(
        default=Path("logs"),
        description="Directory to save log files"
    )
    
    # 2. Data source configuration (now more flexible)
    # A. API-Driven Data Loading & Filtering (New)
    master_data_path: Optional[Path] = Field(
        default=Path("data/master/master_food_prices.parquet"),
        description="Path to the master Parquet file for on-demand filtering."
    )
    filter_commodities: Optional[List[str]] = Field(
        default=None,
        description="List of commodities to filter from the master dataset. If None, all are used."
    )
    filter_cities: Optional[List[str]] = Field(
        default=None,
        description="List of cities to filter from the master dataset. If None, all are used."
    )
    filter_years: Optional[List[int]] = Field(
        default=None,
        description="List of years to filter from the master dataset. If None, all are used."
    )
    
    # B. Legacy File Discovery (made optional)
    input_file_pattern: str = Field(
        default="food_prices_consolidated_*.csv",
        description="Pattern to match input CSV files in processed directory"
    )
    
    processed_data_dir: Path = Field(
        default=Path("data/processed"),
        description="Directory containing processed/consolidated data files"
    )
    
    # 3. Feature engineering parameters
    features_to_extract: List[Literal["avg", "cv", "trend"]] = Field(
        default=["avg"],
        description="Features to extract (avg=average, cv=coeff_of_variation, trend=slope)"
    )
    aggregation_freq: Literal["all", "Y", "M"] = Field(
        default="Y",
        description="Temporal aggregation: 'all'=entire period, 'Y'=yearly, 'M'=monthly"
    )
    
    # Local data loading configuration
    local_data_path: Optional[Path] = Field(
        default=None,
        description="Path to local consolidated data file when no DataFrame provided"
    )
    
    # 4. Feature calculation parameters
    min_data_points: int = Field(
        default=10  ,
        description="Minimum number of data points required for feature calculation in a given period."
    )
    trend_method: Literal["linear_regression", "simple_slope"] = Field(
        default="linear_regression",
        description="Method for calculating price trend"
    )
    
    # Feature naming
    feature_name_format: str = Field(
        default="{commodity}_{feature_type}",
        description="Format for feature column names. Available: {commodity}, {feature_type}"
    )
    
    # Scaling configuration
    scaling_methods: List[Literal["standard", "minmax", "robust"]] = Field(
        default=["robust"],
        description="Scaling methods to apply to features"
    )
    
    # Output configuration
    features_output_dir: Path = Field(
        default=Path("data/features"),
        description="Directory to save feature matrices"
    )
    
    export_formats: List[Literal["csv", "excel", "json"]] = Field(
        default=["csv", "excel"],
        description="Export formats for feature matrices"
    )
    
    # Validation methods
    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate logging level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()
    
    @field_validator('min_data_points')
    @classmethod
    def validate_min_data_points(cls, v: int) -> int:
        if v < 10:
            raise ValueError("Minimum data points must be at least 10")
        return v
    
    @field_validator('features_to_extract')
    @classmethod
    def validate_features_to_extract(cls, v: List[str]) -> List[str]:
        valid_features = ["avg", "cv", "trend"]
        if not v:
            raise ValueError("At least one feature must be specified")
        for feature in v:
            if feature not in valid_features:
                raise ValueError(f"Invalid feature '{feature}'. Valid features: {valid_features}")
        if len(v) != len(set(v)):
            raise ValueError("Duplicate features found")
        return v
    
    @field_validator('export_formats')
    @classmethod
    def validate_export_formats(cls, v: List[str]) -> List[str]:
        valid_formats = ["csv", "excel", "json"]
        for fmt in v:
            if fmt not in valid_formats:
                raise ValueError(f"Export format '{fmt}' not supported. Valid: {valid_formats}")
        return v
    
    @field_validator('scaling_methods')
    @classmethod
    def validate_scaling_methods(cls, v: List[str]) -> List[str]:
        valid_methods = ["standard", "minmax", "robust"]
        for method in v:
            if method not in valid_methods:
                raise ValueError(f"Scaling method '{method}' not supported. Valid: {valid_methods}")
        return v
