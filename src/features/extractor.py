"""
Feature Extraction Module

This module handles the extraction of statistical features from time series
food price data for clustering analysis.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
from sklearn.linear_model import LinearRegression

from .config import FeatureEngineeringConfig

logger = logging.getLogger(__name__)

# Allowed commodities for validation
ALLOWED_COMMODITIES = [
    "Beras", "Telur Ayam", "Daging Ayam", "Daging Sapi",
    "Bawang Merah", "Bawang Putih", "Cabai Merah", "Cabai Rawit",
    "Minyak Goreng", "Gula Pasir"
]


class FeatureExtractor:
    """
    Extracts statistical features from time series food price data.
    
    For each city-commodity combination, extracts:
    1. Price Average: Mean price over the time period
    2. Coefficient of Variation: Volatility measure (std/mean)
    3. Price Trend: Linear regression slope (price change over time)
    """
    
    def __init__(self, config: FeatureEngineeringConfig):
        """
        Initialize the feature extractor.
        
        Args:
            config: Configuration object for feature engineering
        """
        self.config = config
    
    def validate_and_discover_commodities(self, df: pd.DataFrame) -> List[str]:
        """
        Validate commodities in the data and return list of valid commodities.
        
        Args:
            df: DataFrame containing commodity data
            
        Returns:
            List of valid commodities found in the data
            
        Raises:
            ValueError: If no valid commodities found or invalid commodities present
        """
        if 'Commodity' not in df.columns:
            raise ValueError("DataFrame must contain 'Commodity' column")
        
        # Get unique commodities from data
        data_commodities = df['Commodity'].unique().tolist()
        
        if not data_commodities:
            raise ValueError("No commodities found in data")
        
        # Check for invalid commodities
        invalid_commodities = [c for c in data_commodities if c not in ALLOWED_COMMODITIES]
        
        if invalid_commodities:
            raise ValueError(
                f"Invalid commodities found: {invalid_commodities}. "
                f"Allowed commodities: {ALLOWED_COMMODITIES}"
            )
        
        logger.info(f"Valid commodities found in data: {data_commodities}")
        return data_commodities
        
    def calculate_price_average(self, price_series: pd.Series) -> float:
        """
        Calculate the mean price over the time period.
        
        Args:
            price_series: Series of prices for a city-commodity combination
            
        Returns:
            Mean price value
        """
        if len(price_series) == 0 or price_series.isnull().all():
            return np.nan
        
        return price_series.mean()
    
    def calculate_coefficient_of_variation(self, price_series: pd.Series) -> float:
        """
        Calculate the coefficient of variation (CV) as a measure of price volatility.
        
        CV = standard_deviation / mean
        
        Args:
            price_series: Series of prices for a city-commodity combination
            
        Returns:
            Coefficient of variation (volatility measure)
        """
        if len(price_series) == 0 or price_series.isnull().all():
            return np.nan
        
        # Remove null values
        clean_prices = price_series.dropna()
        
        if len(clean_prices) < 2:
            return np.nan
        
        mean_price = clean_prices.mean()
        std_price = clean_prices.std()
        
        # Avoid division by zero
        if mean_price == 0:
            return np.nan
        
        cv = std_price / mean_price
        return cv
    
    def calculate_price_trend(
        self, 
        price_series: pd.Series, 
        date_series: pd.Series
    ) -> float:
        """
        Calculate the price trend (slope) over time.
        
        Args:
            price_series: Series of prices for a city-commodity combination
            date_series: Corresponding series of dates
            
        Returns:
            Price trend (slope coefficient)
        """
        if len(price_series) == 0 or price_series.isnull().all():
            return np.nan
        
        # Create DataFrame and remove null values
        df_temp = pd.DataFrame({'price': price_series, 'date': date_series}).dropna()
        
        if len(df_temp) < 3:  # Need at least 3 points for trend
            return np.nan
        
        # Sort by date
        df_temp = df_temp.sort_values('date')
        
        if self.config.trend_method == "linear_regression":
            # Convert dates to numeric (days since first date)
            df_temp['days'] = (df_temp['date'] - df_temp['date'].min()).dt.days
            
            # Fit linear regression
            X = df_temp['days'].values.reshape(-1, 1)
            y = df_temp['price'].values
            
            try:
                model = LinearRegression()
                model.fit(X, y)
                slope = model.coef_[0]
                return slope
            except Exception:
                return np.nan
                
        elif self.config.trend_method == "simple_slope":
            # Simple slope calculation: (last_price - first_price) / days
            first_price = df_temp['price'].iloc[0]
            last_price = df_temp['price'].iloc[-1]
            total_days = (df_temp['date'].iloc[-1] - df_temp['date'].iloc[0]).days
            
            if total_days == 0:
                return np.nan
            
            slope = (last_price - first_price) / total_days
            return slope
        
        else:
            raise ValueError(f"Unknown trend method: {self.config.trend_method}")
    
    def extract_features_for_city_commodity(
        self, 
        commodity_data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Extract configured features for a single city-commodity combination.
        
        Args:
            commodity_data: DataFrame with price data for one city-commodity
            
        Returns:
            Dictionary with extracted features (only those configured)
        """
        if len(commodity_data) < self.config.min_data_points:
            logger.warning(
                f"Insufficient data points: {len(commodity_data)} < {self.config.min_data_points}"
            )
            # Return NaN for all configured features
            return {feature: np.nan for feature in self.config.features_to_extract}
        
        # Sort by date
        commodity_data = commodity_data.sort_values('Date')
        
        # Extract only the configured features
        features = {}
        
        if 'avg' in self.config.features_to_extract:
            features['avg'] = self.calculate_price_average(commodity_data['Price'])
        
        if 'cv' in self.config.features_to_extract:
            features['cv'] = self.calculate_coefficient_of_variation(commodity_data['Price'])
        
        if 'trend' in self.config.features_to_extract:
            features['trend'] = self.calculate_price_trend(
                commodity_data['Price'], 
                commodity_data['Date']
            )
        
        return features
    
    def create_feature_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create the feature matrix from consolidated time series data.
        
        Args:
            df: Consolidated DataFrame with time series data
            
        Returns:
            DataFrame with cities as rows and features as columns
        """
        logger.info("Starting feature matrix creation")
        logger.info(f"Aggregation frequency: {self.config.aggregation_freq}")
        
        # Route to appropriate method based on aggregation frequency
        if self.config.aggregation_freq == "all":
            return self._create_aggregated_features(df)
        else:
            return self._create_temporal_features(df)
    
    def _create_aggregated_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create feature matrix aggregating all data (original behavior).
        
        Args:
            df: Consolidated DataFrame with time series data
            
        Returns:
            DataFrame with cities as rows and features as columns
        """
        logger.info("Creating aggregated features (all data)")
        
        # Validate and discover commodities in the data
        available_commodities = self.validate_and_discover_commodities(df)
        logger.info(f"Will extract features for {len(available_commodities)} commodities: {available_commodities}")
        logger.info(f"Features to extract: {self.config.features_to_extract}")
        
        # Initialize results list
        feature_rows = []
        
        # Get unique cities
        cities = df['City'].unique()
        logger.info(f"Processing {len(cities)} cities")
        
        for city in cities:
            logger.debug(f"Processing city: {city}")
            
            # Filter data for this city
            city_data = df[df['City'] == city].copy()
            
            # Initialize feature row
            feature_row = {'City': city}
            
            # Process each commodity that exists in the data
            for commodity in available_commodities:
                # Filter data for this commodity
                commodity_data = city_data[city_data['Commodity'] == commodity].copy()
                
                # Extract features
                features = self.extract_features_for_city_commodity(commodity_data)
                
                # Add to feature row with proper naming
                for feature_type, value in features.items():
                    col_name = self.config.feature_name_format.format(
                        commodity=commodity, 
                        feature_type=feature_type
                    )
                    feature_row[col_name] = value
                
                # Format features for logging, handling NaN values
                feature_log_parts = []
                for feature_type, value in features.items():
                    if feature_type == 'avg':
                        val_str = f"{value:.0f}" if not np.isnan(value) else 'NaN'
                    else:
                        val_str = f"{value:.3f}" if not np.isnan(value) else 'NaN'
                    feature_log_parts.append(f"{feature_type}={val_str}")
                
                logger.debug(f"  {commodity}: {', '.join(feature_log_parts)}")
            
            feature_rows.append(feature_row)
        
        # Create DataFrame
        feature_matrix = pd.DataFrame(feature_rows)
        
        # Set City as index
        feature_matrix = feature_matrix.set_index('City')
        
        total_features = len(available_commodities) * len(self.config.features_to_extract)
        logger.info(f"Feature matrix created: {feature_matrix.shape} ({total_features} features expected)")
        return feature_matrix
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create feature matrix with temporal aggregation (yearly or monthly).
        
        Args:
            df: Consolidated DataFrame with time series data
            
        Returns:
            DataFrame with cities as rows and temporal features as columns
        """
        logger.info(f"Creating temporal features with {self.config.aggregation_freq} aggregation")
        
        # Validate and discover commodities in the data
        available_commodities = self.validate_and_discover_commodities(df)
        logger.info(f"Will extract features for {len(available_commodities)} commodities: {available_commodities}")
        logger.info(f"Features to extract: {self.config.features_to_extract}")
        
        # Ensure Date column is datetime
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Create time period column based on aggregation frequency
        if self.config.aggregation_freq == "Y":
            df['Period'] = df['Date'].dt.year
            period_format = "{}"
        elif self.config.aggregation_freq == "M":
            df['Period'] = df['Date'].dt.to_period('M')
            period_format = "{}"
        else:
            raise ValueError(f"Unsupported aggregation frequency: {self.config.aggregation_freq}")
        
        # Get all unique periods and cities
        all_periods = sorted(df['Period'].unique())
        all_cities = sorted(df['City'].unique())
        
        logger.info(f"Processing {len(all_cities)} cities across {len(all_periods)} periods: {all_periods}")
        
        # Initialize results list
        feature_rows = []
        
        for city in all_cities:
            logger.debug(f"Processing city: {city}")
            
            # Filter data for this city
            city_data = df[df['City'] == city].copy()
            
            # Initialize feature row
            feature_row = {'City': city}
            
            # Process each commodity and period combination
            for commodity in available_commodities:
                commodity_data = city_data[city_data['Commodity'] == commodity].copy()
                
                for period in all_periods:
                    # Filter data for this period
                    period_data = commodity_data[commodity_data['Period'] == period].copy()
                    
                    # Extract features for this period
                    if len(period_data) >= self.config.min_data_points:
                        features = self.extract_features_for_city_commodity(period_data)
                    else:
                        # Fill with NaN for insufficient data
                        features = {feature: np.nan for feature in self.config.features_to_extract}
                    
                    # Add to feature row with temporal naming
                    for feature_type, value in features.items():
                        col_name = f"{commodity}_{feature_type}_{period_format.format(period)}"
                        feature_row[col_name] = value
            
            feature_rows.append(feature_row)
        
        # Create DataFrame
        feature_matrix = pd.DataFrame(feature_rows)
        
        # Set City as index
        feature_matrix = feature_matrix.set_index('City')
        
        total_features = len(available_commodities) * len(self.config.features_to_extract) * len(all_periods)
        logger.info(f"Temporal feature matrix created: {feature_matrix.shape} ({total_features} features expected)")
        return feature_matrix
    
    def validate_feature_matrix(self, feature_matrix: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate the feature matrix and compute statistical summaries.
        
        Args:
            feature_matrix: DataFrame with extracted features
            
        Returns:
            Dictionary with validation results
        """
        # Separate numeric and non-numeric columns
        numeric_cols = feature_matrix.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_cols = feature_matrix.select_dtypes(exclude=[np.number]).columns.tolist()
        
        validation_results = {
            'basic_info': {
                'total_cities': len(feature_matrix),
                'total_features': len(numeric_cols),
                'metadata_columns': len(non_numeric_cols),
                'numeric_columns': numeric_cols,
                'metadata_columns_list': non_numeric_cols
            },
            'data_quality': {
                'missing_values_total': feature_matrix[numeric_cols].isnull().sum().sum(),
                'missing_values_by_column': feature_matrix[numeric_cols].isnull().sum().to_dict(),
                'complete_cases': len(feature_matrix.dropna(subset=numeric_cols)),
                'incomplete_cases': len(feature_matrix) - len(feature_matrix.dropna(subset=numeric_cols))
            },
            'feature_statistics': {}
        }
        
        # Calculate statistics for each feature type
        for feature_type in ['avg', 'cv', 'trend']:
            type_cols = [col for col in numeric_cols if col.endswith(f'_{feature_type}')]
            
            if type_cols:
                type_data = feature_matrix[type_cols]
                validation_results['feature_statistics'][feature_type] = {
                    'count': len(type_cols),
                    'min': type_data.min().min(),
                    'max': type_data.max().max(),
                    'mean': type_data.mean().mean(),
                    'median': type_data.median().median(),
                    'std': type_data.std().mean(),
                    'missing_rate': type_data.isnull().sum().sum() / (len(type_data) * len(type_cols))
                }
        
        return validation_results
