"""
Validation utilities for data quality checks.
"""

from typing import Dict, Any
import pandas as pd
import numpy as np


def validate_processed_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate the processed dataset and return quality metrics.
    
    Args:
        df: Processed DataFrame to validate
        
    Returns:
        Dictionary containing validation results and quality metrics
    """
    validation_results = {
        'total_rows': len(df),
        'total_cities': df['City'].nunique(),
        'total_commodities': df['Commodity'].nunique(),
        'date_range': {
            'min_date': df['Date'].min(),
            'max_date': df['Date'].max()
        },
        'missing_values': {
            'price_nulls': df['Price'].isnull().sum(),
            'date_nulls': df['Date'].isnull().sum()
        },
        'price_stats': {
            'min_price': df['Price'].min(),
            'max_price': df['Price'].max(),
            'mean_price': df['Price'].mean(),
            'negative_prices': (df['Price'] < 0).sum()
        }
    }
    
    # Data quality checks
    validation_results['quality_issues'] = []
    
    if validation_results['missing_values']['price_nulls'] > 0:
        validation_results['quality_issues'].append(
            f"Found {validation_results['missing_values']['price_nulls']} null prices"
        )
    
    if validation_results['missing_values']['date_nulls'] > 0:
        validation_results['quality_issues'].append(
            f"Found {validation_results['missing_values']['date_nulls']} null dates"
        )
    
    if validation_results['price_stats']['negative_prices'] > 0:
        validation_results['quality_issues'].append(
            f"Found {validation_results['price_stats']['negative_prices']} negative prices"
        )
    
    return validation_results


def validate_file_structure(file_path: str, required_columns: list) -> bool:
    """
    Validate that a file has the required column structure.
    
    Args:
        file_path: Path to the file to validate
        required_columns: List of required column names
        
    Returns:
        True if file structure is valid, False otherwise
    """
    try:
        df = pd.read_excel(file_path, nrows=1)  # Read only first row for efficiency
        return all(col in df.columns for col in required_columns)
    except Exception:
        return False


def check_data_completeness(df: pd.DataFrame, expected_records_per_city: int = 1300) -> Dict[str, Any]:
    """
    Check completeness of data by city and commodity.
    
    Args:
        df: DataFrame with columns ['City', 'Commodity', 'Date', 'Price']
        expected_records_per_city: Expected number of records per city (5 years * 260 days)
        
    Returns:
        Dictionary with completeness statistics
    """
    completeness_stats = {}
    
    # Records per city
    city_counts = df.groupby('City').size()
    completeness_stats['city_completeness'] = {
        'mean_records': city_counts.mean(),
        'min_records': city_counts.min(),
        'max_records': city_counts.max(),
        'cities_below_threshold': (city_counts < expected_records_per_city * 0.8).sum()
    }
    
    # Records per commodity
    commodity_counts = df.groupby('Commodity').size()
    completeness_stats['commodity_completeness'] = {
        'mean_records': commodity_counts.mean(),
        'min_records': commodity_counts.min(),
        'max_records': commodity_counts.max()
    }
    
    # Missing data by city-commodity combination
    city_commodity_counts = df.groupby(['City', 'Commodity']).size()
    expected_per_combination = expected_records_per_city // 10  # 10 commodities
    completeness_stats['combination_completeness'] = {
        'combinations_with_missing_data': (city_commodity_counts < expected_per_combination * 0.8).sum(),
        'total_combinations': len(city_commodity_counts)
    }
    
    return completeness_stats
