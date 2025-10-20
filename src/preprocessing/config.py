"""
Configuration management for data consolidation pipeline.
"""

from typing import List, Literal, Union
from pathlib import Path
from pydantic import BaseModel, Field, field_validator
import json


def load_valid_cities() -> List[str]:
    """
    Load valid cities from city_coordinates.json.
    
    Returns:
        List of valid city names
    """
    try:
        coords_path = Path("data/city_coordinates.json")
        with open(coords_path, 'r', encoding='utf-8') as f:
            coords_data = json.load(f)
        return list(coords_data.keys())
    except Exception as e:
        print(f"Warning: Could not load city coordinates: {e}")
        return []


class ConsolidationConfig(BaseModel):
    """
    Configuration class for data consolidation pipeline.
    
    Uses Pydantic for validation and type checking.
    """
    
    # Input configuration
    input_type: Literal["directory", "zip"] = Field(
        default="directory",
        description="Type of input: 'directory' for folder structure or 'zip' for ZIP file"
    )
    
    # Logging configuration
    enable_file_logging: bool = Field(
        default=False,
        description="Whether to save logs to file (True) or console only (False)"
    )
    
    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR"
    )
    
    # Data selection parameters (empty list = process all available)
    provinces: List[str] = Field(
        default=[],
        description="List of provinces to process. Empty list = process all provinces"
    )
    
    years: List[str] = Field(
        default=[],
        description="List of years to process. Empty list = process all years (2020-2024)"
    )
    
    commodities: List[str] = Field(
        default=[
            "Beras",
            "Telur Ayam", 
            "Daging Ayam",
            "Daging Sapi",
            "Bawang Merah",
            "Bawang Putih",
            "Cabai Merah",
            "Cabai Rawit",
            "Minyak Goreng",
            "Gula Pasir"
        ],
        description="List of commodities to include in analysis"
    )
    
    # Data processing parameters
    columns_to_drop: List[str] = Field(
        default=["No"],
        description="Column names to drop from raw data"
    )
    
    missing_value_indicators: List[str] = Field(
        default=["-", "", "nan", "NaN", "null", "NULL"],
        description="Values to treat as missing/null"
    )
    
    # File processing parameters
    date_format: str = Field(
        default="%d/ %m/ %Y",
        description="Expected date format in Excel files"
    )
    
    commodity_column_name: str = Field(
        default="Komoditas (Rp)",
        description="Original name of commodity column in Excel files"
    )
    
    # Validation parameters
    valid_cities: List[str] = Field(
        default_factory=load_valid_cities,
        description="List of valid cities (from city_coordinates.json)"
    )
    
    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate logging level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()
    
    @field_validator('years')
    @classmethod
    def validate_years(cls, v: List[str]) -> List[str]:
        """Validate year format and range."""
        # If empty, it will be populated with all available years later
        if not v:
            return v
            
        # Validate year format if provided
        for year in v:
            try:
                year_int = int(year)
                if not (2020 <= year_int <= 2024):
                    raise ValueError(f"Year {year} outside expected range 2020-2024")
            except ValueError as e:
                if "invalid literal" in str(e):
                    raise ValueError(f"Invalid year format: {year}")
                raise
        return v
    
    @field_validator('commodities')
    @classmethod
    def validate_commodities(cls, v: List[str]) -> List[str]:
        """Validate commodities list."""
        if not v:
            raise ValueError("At least one commodity must be specified")
        if len(v) != len(set(v)):
            raise ValueError("Duplicate commodities found")
        return v
