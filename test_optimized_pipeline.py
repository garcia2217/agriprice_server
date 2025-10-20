#!/usr/bin/env python3
"""
Test script for optimized preprocessing pipeline.

This script tests the optimized preprocessing pipeline with data_sumatera.zip
and measures performance improvements.
"""

import time
import psutil
import pandas as pd
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append('src')

from src.preprocessing import ConsolidationConfig, DataConsolidator

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def test_optimized_pipeline():
    """Test the optimized preprocessing pipeline."""
    print("Testing Optimized Preprocessing Pipeline")
    print("=" * 50)
    
    # Test data path
    zip_path = Path("data/zip/data_sumatera.zip")
    
    if not zip_path.exists():
        print(f"Test data not found: {zip_path}")
        return False
    
    print(f"Test data: {zip_path}")
    print(f"File size: {zip_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Configuration
    config = ConsolidationConfig(
        input_type="zip",
        enable_file_logging=False,
        log_level="INFO"
    )
    
    # Initialize consolidator
    consolidator = DataConsolidator(config)
    
    # Measure performance
    start_time = time.time()
    start_memory = get_memory_usage()
    
    print("\nStarting optimized data consolidation...")
    print(f"Initial memory usage: {start_memory:.2f} MB")
    
    try:
        # Run consolidation
        consolidated_df = consolidator.consolidate_data(zip_path, remove_outliers=False)
        
        end_time = time.time()
        end_memory = get_memory_usage()
        
        # Calculate metrics
        duration = end_time - start_time
        memory_used = end_memory - start_memory
        peak_memory = end_memory
        
        print(f"\nConsolidation completed successfully!")
        print(f"Total time: {duration:.2f} seconds")
        print(f"Memory used: {memory_used:.2f} MB")
        print(f"Peak memory: {peak_memory:.2f} MB")
        
        # Data validation
        print(f"\nData Summary:")
        print(f"   Rows: {len(consolidated_df):,}")
        print(f"   Cities: {consolidated_df['City'].nunique()}")
        print(f"   Commodities: {consolidated_df['Commodity'].nunique()}")
        print(f"   Date range: {consolidated_df['Date'].min()} to {consolidated_df['Date'].max()}")
        
        # Check for missing values
        missing_prices = consolidated_df['Price'].isna().sum()
        print(f"   Missing prices: {missing_prices}")
        
        # Sample data
        print(f"\nSample data (first 5 rows):")
        print(consolidated_df[['City', 'Commodity', 'Date', 'Price']].head())
        
        # Performance per file estimation
        # This is an estimate since we don't know exact file count from ZIP
        estimated_files = len(consolidated_df['City'].unique()) * len(consolidated_df['Commodity'].unique())
        time_per_file = duration / estimated_files if estimated_files > 0 else 0
        print(f"\nPerformance metrics:")
        print(f"   Estimated files processed: {estimated_files}")
        print(f"   Time per file: {time_per_file:.3f} seconds")
        print(f"   Files per second: {estimated_files/duration:.1f}")
        
        return True
        
    except Exception as e:
        print(f"\nConsolidation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_data_consistency():
    """Test that the optimized pipeline produces consistent results."""
    print("\nTesting Data Consistency")
    print("=" * 30)
    
    # This would require running the old pipeline for comparison
    # For now, we'll just validate the data structure
    zip_path = Path("data/zip/data_sumatera.zip")
    
    if not zip_path.exists():
        print("Test data not found")
        return False
    
    config = ConsolidationConfig(input_type="zip", enable_file_logging=False)
    consolidator = DataConsolidator(config)
    
    try:
        df = consolidator.consolidate_data(zip_path, remove_outliers=False)
        
        # Basic consistency checks
        required_columns = ['City', 'Commodity', 'Date', 'Price', 'Year', 'Month', 'Province', 'Source_File']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
        
        # Check data types
        if not pd.api.types.is_datetime64_any_dtype(df['Date']):
            print("Date column is not datetime type")
            return False
        
        if not pd.api.types.is_numeric_dtype(df['Price']):
            print("Price column is not numeric type")
            return False
        
        # Check for reasonable data ranges
        if df['Price'].min() < 0:
            print("Negative prices found")
            return False
        
        if df['Date'].min().year < 2020 or df['Date'].max().year > 2024:
            print("Date range outside expected bounds")
            return False
        
        print("Data consistency checks passed")
        return True
        
    except Exception as e:
        print(f"Consistency test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("Preprocessing Pipeline Optimization Test")
    print("=" * 50)
    
    # Test 1: Performance test
    success1 = test_optimized_pipeline()
    
    # Test 2: Consistency test
    success2 = test_data_consistency()
    
    print(f"\nTest Results:")
    print(f"   Performance test: {'PASSED' if success1 else 'FAILED'}")
    print(f"   Consistency test: {'PASSED' if success2 else 'FAILED'}")
    
    if success1 and success2:
        print(f"\nAll tests passed! The optimized pipeline is working correctly.")
    else:
        print(f"\nSome tests failed. Please check the implementation.")
