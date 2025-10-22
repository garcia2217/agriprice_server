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
import re


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
        errors.append("No file provided in request. Please upload a ZIP file containing Excel data.")
        return False, errors
    
    # Check file extension
    if not file.name.lower().endswith('.zip'):
        errors.append(f"File '{file.name}' must be a ZIP archive. Please compress your Excel files into a ZIP file.")
        return False, errors
    
    # Check file size (reasonable limit)
    max_size = 50 * 1024 * 1024  # 50MB
    if file.size > max_size:
        size_mb = file.size / 1024 / 1024
        errors.append(f"File size {size_mb:.1f}MB exceeds maximum 50MB. Please reduce file size or split into smaller archives.")
        return False, errors
    
    # Try to open as ZIP
    try:
        file.seek(0)
        with zipfile.ZipFile(file, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            if not file_list:
                errors.append("ZIP file is empty. Please ensure the ZIP contains Excel files with food price data.")
                return False, errors
            
            # Check for Excel files
            excel_files = [f for f in file_list if f.lower().endswith(('.xlsx', '.xls'))]
            if not excel_files:
                errors.append("No Excel files found in ZIP. Please include .xlsx or .xls files with food price data.")
                return False, errors
                
    except zipfile.BadZipFile:
        errors.append("Invalid ZIP file format. The file appears to be corrupted or not a valid ZIP archive.")
        return False, errors
    except Exception as e:
        errors.append(f"Error reading ZIP file: {str(e)}. Please ensure the file is not corrupted.")
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
        errors.append("City column not found in data. Please ensure your Excel files contain a 'City' column.")
        return False, errors
    
    # Get unique cities in the data
    data_cities = set(df['City'].unique())
    valid_cities_set = set(valid_cities)
    
    # Find invalid cities
    invalid_cities = data_cities - valid_cities_set
    
    if invalid_cities:
        # Show first 5 invalid cities and total count
        invalid_list = sorted(list(invalid_cities))
        display_list = invalid_list[:5]
        if len(invalid_list) > 5:
            display_list.append(f"... and {len(invalid_list) - 5} more")
        
        errors.append(
            f"Invalid cities found: {display_list}. "
            f"Supported cities include: {sorted(valid_cities)[:10]}{'...' if len(valid_cities) > 10 else ''}. "
            f"Please check city names in your data."
        )
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
        available_columns = list(df.columns)
        errors.append(
            f"Missing required columns: {missing_columns}. "
            f"Available columns: {available_columns}. "
            f"Please ensure your Excel files contain columns for City, Commodity, Date, and Price."
        )
        return False, errors
    
    # Check data types
    if 'Price' in df.columns:
        if not pd.api.types.is_numeric_dtype(df['Price']):
            errors.append(
                "Price column must contain numeric values. "
                "Please check for non-numeric characters or formatting issues in the Price column."
            )
            return False, errors
        
        # Check for negative prices
        negative_count = (df['Price'] < 0).sum()
        if negative_count > 0:
            errors.append(
                f"Found {negative_count} negative price values. "
                f"Price values cannot be negative. Please check your data for errors."
            )
            return False, errors
    
    # Check for minimum data requirements
    if len(df) < 100:
        errors.append(
            f"Insufficient data: only {len(df)} records found (minimum 100 required). "
            f"Please ensure your Excel files contain enough data for analysis."
        )
        return False, errors
    
    # Check date column
    if 'Date' in df.columns:
        try:
            df_copy = df.copy()
            df_copy['Date'] = pd.to_datetime(df_copy['Date'])
            date_range = (df_copy['Date'].max() - df_copy['Date'].min()).days
            if date_range < 30:
                errors.append(
                    f"Insufficient date range: {date_range} days (minimum 30 days required). "
                    f"Please ensure your data covers at least 30 days for meaningful analysis."
                )
                return False, errors
        except Exception as e:
            errors.append(
                f"Invalid date format: {str(e)}. "
                f"Please ensure dates are in a recognizable format (e.g., DD/MM/YYYY or MM/DD/YYYY)."
            )
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
        errors.append("Commodity column not found in data. Please ensure your Excel files contain a 'Commodity' column.")
        return False, errors
    
    data_commodities = set(df['Commodity'].unique())
    supported_set = set(supported_commodities)
    
    # Check if all commodities are supported
    unsupported = data_commodities - supported_set
    if unsupported:
        unsupported_list = sorted(list(unsupported))
        supported_list = sorted(supported_commodities)
        errors.append(
            f"Unsupported commodities found: {unsupported_list}. "
            f"Supported commodities: {supported_list}. "
            f"Please use only the supported commodity names in your data."
        )
        return False, errors
    
    # Check minimum commodity count for radar chart
    if len(data_commodities) < 3:
        errors.append(
            f"Insufficient commodities: {len(data_commodities)} found (minimum 3 required for radar chart). "
            f"Please ensure your data contains at least 3 different commodities."
        )
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


def validate_zip_structure(file) -> Tuple[bool, List[str]]:
    """
    Validate ZIP file structure: PROVINSI → KOTA → YYYY.xlsx
    
    Args:
        file: Uploaded ZIP file object
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    try:
        # Load valid cities
        valid_cities = load_valid_cities()
        
        # Reset file pointer
        file.seek(0)
        
        with zipfile.ZipFile(file, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            
            # Parse ZIP structure
            structure = {}
            excel_files = []
            
            for file_path in file_list:
                # Skip directories
                if file_path.endswith('/'):
                    continue
                    
                parts = file_path.split('/')
                
                # Handle both structures:
                # 1. Direct: PROVINSI/KOTA/YYYY.xlsx (len=3)
                # 2. With root folder: ROOT/PROVINSI/KOTA/YYYY.xlsx (len=4)
                if len(parts) == 3:
                    provinsi, kota, excel_file = parts
                elif len(parts) == 4:
                    root_folder, provinsi, kota, excel_file = parts
                else:
                    errors.append(f"Invalid file structure: {file_path}. Expected: PROVINSI/KOTA/YYYY.xlsx or ROOT/PROVINSI/KOTA/YYYY.xlsx")
                    continue
                
                # Validate Excel file format
                if excel_file.lower().endswith('.xlsx'):
                    # Check year format (YYYY.xlsx)
                    year_match = re.match(r'^(\d{4})\.xlsx$', excel_file, re.IGNORECASE)
                    if year_match:
                        year = int(year_match.group(1))
                        excel_files.append({
                            'provinsi': provinsi,
                            'kota': kota,
                            'file': excel_file,
                            'year': year,
                            'path': file_path
                        })
                    else:
                        errors.append(f"Invalid Excel filename '{excel_file}' in {file_path}. Expected format: YYYY.xlsx")
                else:
                    errors.append(f"Non-Excel file found: {file_path}. Only .xlsx files are allowed.")
            
            # Check if we have any valid Excel files
            if not excel_files:
                errors.append("No valid Excel files found in ZIP. Expected structure: PROVINSI/KOTA/YYYY.xlsx or ROOT/PROVINSI/KOTA/YYYY.xlsx")
                return False, errors
            
            # Validate city names
            cities_found = set()
            for excel_info in excel_files:
                kota = excel_info['kota']
                cities_found.add(kota)
                
                if kota not in valid_cities:
                    errors.append(f"Unknown city '{kota}' found. Valid cities: {', '.join(sorted(valid_cities))}")
            
            # Check for duplicate cities
            # if len(cities_found) != len([info['kota'] for info in excel_files]):
            #     errors.append("Duplicate city folders found in ZIP structure.")
            
            # Validate Excel content structure
            for excel_info in excel_files:
                try:
                    # Read Excel file from ZIP
                    with zip_ref.open(excel_info['path']) as excel_file:
                        df = pd.read_excel(excel_file)
                        
                        # Check required columns
                        required_columns = ['No', 'Komoditas (Rp)']
                        missing_columns = [col for col in required_columns if col not in df.columns]
                        
                        if missing_columns:
                            errors.append(f"Missing required columns in {excel_info['path']}: {missing_columns}")
                            continue
                        
                        # Check for date columns (should be in DD/ MM/ YYYY format)
                        date_columns = []
                        for col in df.columns:
                            if re.match(r'^\d{2}/\s*\d{2}/\s*\d{4}$', str(col)):
                                date_columns.append(col)
                        
                        if not date_columns:
                            errors.append(f"No date columns found in {excel_info['path']}. Expected format: DD/ MM/ YYYY")
                            continue
                        
                        # Check if we have price data (numeric values in date columns)
                        has_price_data = False
                        for col in date_columns:
                            # Check if column has numeric values (excluding NaN and '-')
                            df[col] = (
                                df[col]
                                .astype(str)
                                .str.replace(',', '', regex=False)
                                .str.strip()
                                .replace('', None)
                            )
                            numeric_values = pd.to_numeric(df[col], errors='coerce')
                            if not numeric_values.isna().all():
                                has_price_data = True
                                break
                        
                        if not has_price_data:
                            errors.append(f"No price data found in {excel_info['path']}. All date columns appear to be empty or contain only non-numeric values.")
                        
                        # Check commodity data
                        if df['Komoditas (Rp)'].isna().all():
                            errors.append(f"No commodity data found in {excel_info['path']}. 'Komoditas (Rp)' column is empty.")
                        
                except Exception as e:
                    errors.append(f"Error reading Excel file {excel_info['path']}: {str(e)}")
            
            # If we have errors, return False
            if errors:
                return False, errors
            
            # Success - log structure info
            print(f"✅ ZIP structure validation passed:")
            print(f"   - Found {len(excel_files)} Excel files")
            print(f"   - Cities: {', '.join(sorted(cities_found))}")
            print(f"   - Years: {', '.join(sorted(set(str(info['year']) for info in excel_files)))}")
            print(f"   - Structure: PROVINSI/KOTA/YYYY.xlsx (with optional root folder)")
            
            return True, errors
            
    except zipfile.BadZipFile:
        errors.append("Invalid ZIP file format. The file appears to be corrupted or not a valid ZIP archive.")
        return False, errors
    except Exception as e:
        errors.append(f"Error validating ZIP structure: {str(e)}")
        return False, errors