"""
Data cleaning functionality for food price data.
"""

import logging
from typing import Dict, Any
import pandas as pd
import numpy as np

from .config import ConsolidationConfig

logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Handles data cleaning operations including missing values, data types, and filtering.
    """
    
    def __init__(self, config: ConsolidationConfig):
        """
        Initialize DataCleaner with configuration.
        
        Args:
            config: Configuration object with cleaning parameters
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def clean_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values using forward/backward fill within city-commodity groups.
        
        Args:
            df: DataFrame with potential missing values
            
        Returns:
            DataFrame with missing values handled
        """
        df_clean = df.copy()
        
        # Replace missing value indicators with NaN
        for indicator in self.config.missing_value_indicators:
            df_clean["Price"] = df_clean["Price"].replace(indicator, np.nan)
        
        # Sort for proper forward/backward fill
        df_clean = df_clean.sort_values(["City", "Commodity", "Date"])
        
        # Forward fill then backward fill within groups
        df_clean["Price"] = (
            df_clean.groupby(["City", "Commodity"])["Price"]
            .ffill()
            .bfill()
        )
        
        return df_clean
    
    def convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert columns to appropriate data types.
        
        Args:
            df: DataFrame to convert
            
        Returns:
            DataFrame with converted data types
        """
        df_converted = df.copy()
        
        # Convert price column
        # Remove commas and convert to numeric
        df_converted["Price"] = pd.to_numeric(
            df_converted["Price"].astype(str).str.replace(",", "", regex=False),
            errors="coerce"
        )
        
        # Convert date column
        df_converted["Date"] = pd.to_datetime(
            df_converted["Date"],
            format=self.config.date_format,
            errors="coerce"
        )
        
        # Convert categorical columns for memory efficiency
        categorical_columns = ["City", "Commodity", "Year", "Province", "Source_File"]
        for col in categorical_columns:
            if col in df_converted.columns:
                df_converted[col] = df_converted[col].astype("category")
        
        return df_converted
    
    def filter_commodities(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter DataFrame to include only specified commodities.
        
        Args:
            df: DataFrame to filter
            
        Returns:
            Filtered DataFrame
        """
        return df[df["Commodity"].isin(self.config.commodities)].copy()
    
    def remove_outliers(self, df: pd.DataFrame, method: str = "iqr", threshold: float = 1.5) -> pd.DataFrame:
        """
        Remove price outliers using specified method.
        
        Args:
            df: DataFrame with price data
            method: Outlier detection method ('iqr' or 'zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outliers removed
        """
        df_clean = df.copy()
        
        if method == "iqr":
            # Remove outliers using IQR method per commodity
            for commodity in df_clean["Commodity"].unique():
                commodity_mask = df_clean["Commodity"] == commodity
                prices = df_clean.loc[commodity_mask, "Price"]
                
                Q1 = prices.quantile(0.25)
                Q3 = prices.quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                # Mark outliers
                outlier_mask = (prices < lower_bound) | (prices > upper_bound)
                df_clean = df_clean.loc[~(commodity_mask & outlier_mask)]
                
                if outlier_mask.sum() > 0:
                    self.logger.info(f"Removed {outlier_mask.sum()} outliers for {commodity}")
        
        elif method == "zscore":
            # Remove outliers using Z-score method per commodity
            for commodity in df_clean["Commodity"].unique():
                commodity_mask = df_clean["Commodity"] == commodity
                prices = df_clean.loc[commodity_mask, "Price"]
                
                z_scores = np.abs((prices - prices.mean()) / prices.std())
                outlier_mask = z_scores > threshold
                
                df_clean = df_clean.loc[~(commodity_mask & outlier_mask)]
                
                if outlier_mask.sum() > 0:
                    self.logger.info(f"Removed {outlier_mask.sum()} outliers for {commodity}")
        
        return df_clean
    
    def validate_date_range(self, df: pd.DataFrame, start_year: int = 2020, end_year: int = 2024) -> pd.DataFrame:
        """
        Filter data to valid date range and remove invalid dates.
        
        Args:
            df: DataFrame with date column
            start_year: Minimum valid year
            end_year: Maximum valid year
            
        Returns:
            DataFrame with valid dates only
        """
        df_clean = df.copy()
        
        # Remove rows with invalid dates
        invalid_dates = df_clean["Date"].isna()
        if invalid_dates.sum() > 0:
            self.logger.warning(f"Removing {invalid_dates.sum()} rows with invalid dates")
            df_clean = df_clean.dropna(subset=["Date"])
        
        # Filter to valid year range
        start_date = pd.Timestamp(f"{start_year}-01-01")
        end_date = pd.Timestamp(f"{end_year}-12-31")
        
        date_mask = (df_clean["Date"] >= start_date) & (df_clean["Date"] <= end_date)
        invalid_range = (~date_mask).sum()
        
        if invalid_range > 0:
            self.logger.warning(f"Removing {invalid_range} rows outside date range {start_year}-{end_year}")
            df_clean = df_clean.loc[date_mask]
        
        return df_clean
    
    def clean_dataframe(self, df: pd.DataFrame, remove_outliers: bool = False) -> pd.DataFrame:
        """
        Apply all cleaning operations to a DataFrame.
        
        Args:
            df: Raw DataFrame to clean
            remove_outliers: Whether to remove price outliers
            
        Returns:
            Fully cleaned DataFrame
        """
        self.logger.info("Starting data cleaning process")
        
        # Step 1: Clean missing values
        df_clean = self.clean_missing_values(df)
        self.logger.info("Completed missing value cleaning")
        
        # Step 2: Convert data types
        df_clean = self.convert_data_types(df_clean)
        self.logger.info("Completed data type conversion")
        
        # Step 3: Filter commodities
        df_clean = self.filter_commodities(df_clean)
        self.logger.info("Completed commodity filtering")
        
        # Step 4: Validate date range
        df_clean = self.validate_date_range(df_clean)
        self.logger.info("Completed date validation")
        
        # Step 5: Remove outliers (optional)
        if remove_outliers:
            df_clean = self.remove_outliers(df_clean)
            self.logger.info("Completed outlier removal")
        
        self.logger.info(f"Cleaning completed. Final shape: {df_clean.shape}")
        return df_clean
