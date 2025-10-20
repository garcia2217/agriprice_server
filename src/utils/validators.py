"""
Data validation utilities for the clustering analysis application.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from pathlib import Path
import json
import zipfile
import io


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


def validate_file_format(file) -> Tuple[bool, List[str]]:
    """
    Validate if uploaded file is a valid ZIP format.
    
    Args:
        file: Uploaded file object
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    if not file:
        errors.append("No file provided")
        return False, errors
    
    # Check file extension
    if not file.name.lower().endswith('.zip'):
        errors.append("File must be a ZIP archive")
        return False, errors
    
    # Check file size (reasonable limit)
    max_size = 50 * 1024 * 1024  # 50MB
    if file.size > max_size:
        errors.append(f"File too large: {file.size / 1024 / 1024:.1f}MB (max 50MB)")
        return False, errors
    
    # Try to open as ZIP
    try:
        file.seek(0)
        with zipfile.ZipFile(file, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            if not file_list:
                errors.append("ZIP file is empty")
                return False, errors
    except zipfile.BadZipFile:
        errors.append("Invalid ZIP file format")
        return False, errors
    except Exception as e:
        errors.append(f"Error reading ZIP file: {str(e)}")
        return False, errors
    
    return True, errors


def validate_geographic_scope(df: pd.DataFrame, valid_cities: List[str]) -> Tuple[bool, List[str]]:
    """
    Validate that all cities in the dataframe are in the valid cities list.
    
    Args:
        df: Consolidated dataframe
        valid_cities: List of valid city names
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    if 'City' not in df.columns:
        errors.append("City column not found in data")
        return False, errors
    
    # Get unique cities in the data
    data_cities = set(df['City'].unique())
    valid_cities_set = set(valid_cities)
    
    # Find invalid cities
    invalid_cities = data_cities - valid_cities_set
    
    if invalid_cities:
        errors.append(f"Invalid cities found: {sorted(list(invalid_cities))} - not in supported scope")
        return False, errors
    
    return True, errors


def validate_data_structure(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate that the dataframe has the required structure and columns.
    
    Args:
        df: Dataframe to validate
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Required columns
    required_columns = ['City', 'Commodity', 'Date', 'Price']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")
        return False, errors
    
    # Check data types
    if 'Price' in df.columns:
        if not pd.api.types.is_numeric_dtype(df['Price']):
            errors.append("Price column must be numeric")
            return False, errors
        
        # Check for negative prices
        if (df['Price'] < 0).any():
            errors.append("Price values cannot be negative")
            return False, errors
    
    # Check for minimum data requirements
    if len(df) < 100:
        errors.append(f"Insufficient data: only {len(df)} records found (minimum 100)")
        return False, errors
    
    # Check date column
    if 'Date' in df.columns:
        try:
            df['Date'] = pd.to_datetime(df['Date'])
            date_range = (df['Date'].max() - df['Date'].min()).days
            if date_range < 30:
                errors.append(f"Insufficient date range: {date_range} days (minimum 30 days)")
                return False, errors
        except Exception as e:
            errors.append(f"Invalid date format: {str(e)}")
            return False, errors
    
    return True, errors


def validate_data_quality(df: pd.DataFrame) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Validate data quality and calculate quality metrics.
    
    Args:
        df: Dataframe to validate
        
    Returns:
        Tuple of (is_valid, warnings, quality_metrics)
    """
    warnings = []
    quality_metrics = {}
    
    # Calculate missing data percentage
    missing_percentage = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    quality_metrics['missing_data_percentage'] = round(missing_percentage, 2)
    
    if missing_percentage > 50:
        warnings.append(f"High missing data: {missing_percentage:.1f}%")
    
    # Check for duplicates
    if 'City' in df.columns and 'Commodity' in df.columns and 'Date' in df.columns:
        duplicates = df.duplicated(subset=['City', 'Commodity', 'Date']).sum()
        quality_metrics['duplicate_records'] = int(duplicates)
        
        if duplicates > 0:
            warnings.append(f"Found {duplicates} duplicate records")
    
    # Check price outliers
    if 'Price' in df.columns:
        prices = df['Price'].dropna()
        if len(prices) > 0:
            Q1 = prices.quantile(0.25)
            Q3 = prices.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = prices[(prices < lower_bound) | (prices > upper_bound)]
            outlier_percentage = (len(outliers) / len(prices)) * 100
            quality_metrics['outlier_percentage'] = round(outlier_percentage, 2)
            
            if outlier_percentage > 10:
                warnings.append(f"High outlier percentage: {outlier_percentage:.1f}%")
    
    # Calculate data completeness per commodity
    if 'Commodity' in df.columns and 'Price' in df.columns:
        commodity_completeness = {}
        for commodity in df['Commodity'].unique():
            commodity_data = df[df['Commodity'] == commodity]
            completeness = (1 - commodity_data['Price'].isnull().sum() / len(commodity_data)) * 100
            commodity_completeness[commodity] = round(completeness, 2)
            
            if completeness < 50:
                warnings.append(f"Low data completeness for {commodity}: {completeness:.1f}%")
        
        quality_metrics['commodity_completeness'] = commodity_completeness
    
    # Calculate overall quality score
    quality_score = 1.0
    if missing_percentage > 0:
        quality_score -= (missing_percentage / 100) * 0.3
    if outlier_percentage > 0:
        quality_score -= (outlier_percentage / 100) * 0.2
    
    quality_metrics['score'] = max(0.0, min(1.0, quality_score))
    quality_metrics['data_completeness'] = 1 - (missing_percentage / 100)
    
    return True, warnings, quality_metrics


def extract_available_options(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Extract available commodities, years, cities, and provinces from validated data.
    
    Args:
        df: Validated dataframe
        
    Returns:
        Dictionary with available options
    """
    options = {}
    
    # Extract commodities
    if 'Commodity' in df.columns:
        options['commodities'] = sorted(df['Commodity'].unique().tolist())
    
    # Extract years
    if 'Date' in df.columns:
        try:
            df_copy = df.copy()
            df_copy['Date'] = pd.to_datetime(df_copy['Date'])
            years = sorted(df_copy['Date'].dt.year.unique().tolist())
            options['years'] = years
        except:
            options['years'] = []
    
    # Extract cities and provinces with nested structure
    if 'City' in df.columns and 'Province' in df.columns:
        # Create nested structure: province -> [cities]
        province_city_map = {}
        for _, row in df[['Province', 'City']].drop_duplicates().iterrows():
            province = row['Province']
            city = row['City']
            if province not in province_city_map:
                province_city_map[province] = []
            if city not in province_city_map[province]:
                province_city_map[province].append(city)
        
        # Sort cities within each province
        for province in province_city_map:
            province_city_map[province].sort()
        
        options['provinces'] = province_city_map
        options['cities'] = sorted(df['City'].unique().tolist())
    elif 'City' in df.columns:
        options['cities'] = sorted(df['City'].unique().tolist())
        options['provinces'] = {}
    else:
        options['cities'] = []
        options['provinces'] = {}
    
    # Date range
    if 'Date' in df.columns:
        try:
            df_copy = df.copy()
            df_copy['Date'] = pd.to_datetime(df_copy['Date'])
            options['date_range'] = {
                'start': df_copy['Date'].min().strftime('%Y-%m-%d'),
                'end': df_copy['Date'].max().strftime('%Y-%m-%d')
            }
        except:
            options['date_range'] = {'start': None, 'end': None}
    
    # Total records
    options['total_records'] = len(df)
    
    return options


def calculate_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate comprehensive data summary statistics.
    
    Args:
        df: Dataframe to analyze
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {}
    
    # Basic statistics
    summary['total_records'] = len(df)
    summary['total_cities'] = df['City'].nunique() if 'City' in df.columns else 0
    summary['total_commodities'] = df['Commodity'].nunique() if 'Commodity' in df.columns else 0
    
    # Date range
    if 'Date' in df.columns:
        try:
            df_copy = df.copy()
            df_copy['Date'] = pd.to_datetime(df_copy['Date'])
            summary['date_range_days'] = (df_copy['Date'].max() - df_copy['Date'].min()).days
        except:
            summary['date_range_days'] = 0
    
    # Price statistics
    if 'Price' in df.columns:
        prices = df['Price'].dropna()
        if len(prices) > 0:
            summary['price_stats'] = {
                'min': float(prices.min()),
                'max': float(prices.max()),
                'mean': float(prices.mean()),
                'median': float(prices.median()),
                'std': float(prices.std())
            }
    
    return summary


def validate_commodities(df: pd.DataFrame, supported_commodities: List[str]) -> Tuple[bool, List[str]]:
    """
    Validate that commodities in the data are supported.
    
    Args:
        df: Dataframe to validate
        supported_commodities: List of supported commodity names
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    if 'Commodity' not in df.columns:
        errors.append("Commodity column not found")
        return False, errors
    
    data_commodities = set(df['Commodity'].unique())
    supported_set = set(supported_commodities)
    
    # Check if all commodities are supported
    unsupported = data_commodities - supported_set
    if unsupported:
        errors.append(f"Unsupported commodities found: {sorted(list(unsupported))}")
        return False, errors
    
    # Check minimum commodity count for radar chart
    if len(data_commodities) < 3:
        errors.append(f"Insufficient commodities: {len(data_commodities)} found (minimum 3 for radar chart)")
        return False, errors
    
    return True, errors


def validate_temporal_range(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate temporal range of the data.
    
    Args:
        df: Dataframe to validate
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    if 'Date' not in df.columns:
        errors.append("Date column not found")
        return False, errors
    
    try:
        df_copy = df.copy()
        df_copy['Date'] = pd.to_datetime(df_copy['Date'])
        years = df_copy['Date'].dt.year.unique()
        
        # Check year range (2020-2024)
        valid_years = set(range(2020, 2025))
        invalid_years = set(years) - valid_years
        
        if invalid_years:
            errors.append(f"Invalid years found: {sorted(list(invalid_years))} (supported range: 2020-2024)")
            return False, errors
        
    except Exception as e:
        errors.append(f"Error validating temporal range: {str(e)}")
        return False, errors
    
    return True, errors