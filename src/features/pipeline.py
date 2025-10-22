"""
Feature Engineering Pipeline

This module orchestrates the complete feature engineering pipeline,
from loading consolidated data to extracting features and scaling them.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union
from datetime import datetime
import pandas as pd

from .config import FeatureEngineeringConfig
from .extractor import FeatureExtractor
from .scaler import FeatureScaler
from src.utils.logger import setup_logging

logger = logging.getLogger(__name__)


class FeatureEngineeringPipeline:
    """
    Orchestrates the complete feature engineering pipeline.
    
    Handles:
    1. Loading consolidated time series data
    2. Extracting statistical features (avg, cv, trend)
    3. Scaling features using multiple methods
    4. Exporting both scaled and unscaled feature matrices
    5. Validation and quality checks
    """
    
    def __init__(self, config: FeatureEngineeringConfig):
        """
        Initialize the feature engineering pipeline.
        
        Args:
            config: Configuration object for feature engineering
        """
        self.config = config
        self.extractor = FeatureExtractor(config)
        self.scaler = FeatureScaler(config)
        
        # Setup logging
        setup_logging(
            enable_file_logging=config.enable_file_logging,
            log_level=config.log_level,
            log_dir=config.log_dir
        )
        
    def _load_and_filter_data(self) -> Optional[pd.DataFrame]:
        """Loads and filters data from the master Parquet file based on config.
        
        Returns:
            A DataFrame with the filtered data, or None if the master file is not found.
            
        Raises:
            FileNotFoundError: If the configured master_data_path does not exist.
        """
        if not self.config.master_data_path:
            logger.warning("`master_data_path` is not configured.")
            return None
        
        if not self.config.master_data_path.exists():
            raise FileNotFoundError(f"Master data file not found at: {self.config.master_data_path}")
            
        logger.info(f"Loading master data from: {self.config.master_data_path}")
        df = pd.read_parquet(self.config.master_data_path)
        filtered_df = self._filter_data(df=df)
            
        return filtered_df
    
    def _filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # ensure date column is in datetime type
        df["Date"] = pd.to_datetime(df["Date"])
        
        # Apply filters from config
        initial_rows = len(df)
        if self.config.filter_years:
            df = df[df['Date'].dt.year.isin(self.config.filter_years)]
        if self.config.filter_cities:
            df = df[df['City'].isin(self.config.filter_cities)]
        if self.config.filter_commodities:
            df = df[df['Commodity'].isin(self.config.filter_commodities)]
            
        return df
    
    def find_latest_consolidated_file(self) -> Optional[Path]:
        """Finds the most recent consolidated data file matching the pattern."""
        if not self.config.processed_data_dir or not self.config.input_file_pattern:
            logger.warning("Legacy file discovery paths are not configured.")
            return None
            
        processed_dir = self.config.processed_data_dir
        pattern = self.config.input_file_pattern
        matching_files = sorted(
            processed_dir.glob(pattern), 
            key=lambda f: f.stat().st_mtime, 
            reverse=True
        )
        
        if not matching_files:
            logger.error(f"No consolidated data files found matching pattern: {pattern}")
            return None
        
        latest_file = matching_files[0]
        logger.info(f"Found latest consolidated data file: {latest_file}")
        return latest_file
    
    def load_consolidated_data(self, file_path: Path) -> pd.DataFrame:
        """Loads consolidated data from CSV or Parquet with proper data types."""
        logger.info(f"Loading consolidated data from: {file_path}")
        
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix == '.parquet':
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

        # Ensure correct data types
        df['Date'] = pd.to_datetime(df['Date'])
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        
        categorical_columns = ['City', 'Commodity', 'Year']
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].astype('category')
        
        logger.info(f"Data loaded successfully: {df.shape}")
        return df
    
    def run_feature_extraction(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run feature extraction on consolidated data.
        
        Args:
            df: Consolidated DataFrame with time series data
            
        Returns:
            DataFrame with extracted features (unscaled)
        """
        logger.info("Starting feature extraction pipeline")
        
        # Extract features
        feature_matrix = self.extractor.create_feature_matrix(df)
        
        # Validate features
        validation_results = self.extractor.validate_feature_matrix(feature_matrix)
        
        # Log validation results
        basic_info = validation_results['basic_info']
        quality = validation_results['data_quality']
        
        logger.info(f"Feature extraction completed:")
        logger.info(f"  Cities: {basic_info['total_cities']}")
        logger.info(f"  Features: {basic_info['total_features']}")
        logger.info(f"  Missing values: {quality['missing_values_total']}")
        logger.info(f"  Complete cases: {quality['complete_cases']}")
        
        return feature_matrix
    
    def run_feature_scaling(
        self, 
        feature_matrix: pd.DataFrame
    ) -> Dict[str, Tuple[pd.DataFrame, Any]]:
        """
        Run feature scaling using all configured methods.
        
        Args:
            feature_matrix: Unscaled feature matrix
            
        Returns:
            Dictionary mapping method names to (scaled_dataframe, scaler) tuples
        """
        logger.info("Starting feature scaling pipeline")
        
        # Scale using all methods
        scaled_results = self.scaler.scale_all_methods(feature_matrix)
        
        logger.info(f"Feature scaling completed for {len(scaled_results)} methods")
        return scaled_results
    
    def export_all_features(
        self, 
        unscaled_features: pd.DataFrame,
        scaled_results: Dict[str, Tuple[pd.DataFrame, Any]],
        timestamp: str = None
    ) -> Dict[str, Any]:
        """
        Export both unscaled and scaled feature matrices.
        
        Args:
            unscaled_features: Unscaled feature matrix
            scaled_results: Dictionary of scaling results
            timestamp: Optional timestamp for filenames
            
        Returns:
            Dictionary with all export paths
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info("Exporting feature matrices")
        
        output_dir = self.config.features_output_dir
        
        # Export unscaled features
        unscaled_paths = self.scaler.export_unscaled_features(
            unscaled_features, 
            output_dir, 
            timestamp
        )
        
        # Export scaled features
        scaled_paths = self.scaler.export_scaled_features(
            scaled_results, 
            output_dir, 
            timestamp
        )
        
        export_results = {
            'unscaled': unscaled_paths,
            'scaled': scaled_paths,
            'timestamp': timestamp,
            'output_directory': str(output_dir)
        }
        
        logger.info("Feature matrix export completed")
        return export_results
    
    def _load_and_filter_local_data(self) -> Optional[pd.DataFrame]:
        """
        Load and filter data from local_data_path based on configuration filters.
        
        Returns:
            Filtered DataFrame or None if local_data_path not configured
            
        Raises:
            FileNotFoundError: If local_data_path does not exist
        """
        if not self.config.local_data_path:
            logger.warning("`local_data_path` is not configured.")
            return None
        
        if not self.config.local_data_path.exists():
            raise FileNotFoundError(f"Local data file not found at: {self.config.local_data_path}")
            
        logger.info(f"Loading local data from: {self.config.local_data_path}")
        
        # Load data based on file extension
        if self.config.local_data_path.suffix.lower() == '.csv':
            df = pd.read_csv(self.config.local_data_path)
        elif self.config.local_data_path.suffix.lower() == '.parquet':
            df = pd.read_parquet(self.config.local_data_path)
        else:
            raise ValueError(f"Unsupported file format: {self.config.local_data_path.suffix}")
        
        # Ensure correct data types
        df['Date'] = pd.to_datetime(df['Date'])
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        
        # Apply filters from config
        initial_rows = len(df)
        
        if self.config.filter_years:
            df = df[df['Date'].dt.year.isin(self.config.filter_years)]
            logger.info(f"Filtered by years {self.config.filter_years}: {len(df)} rows remaining")
            
        if self.config.filter_cities:
            df = df[df['City'].isin(self.config.filter_cities)]
            logger.info(f"Filtered by cities {self.config.filter_cities}: {len(df)} rows remaining")
            
        if self.config.filter_commodities:
            df = df[df['Commodity'].isin(self.config.filter_commodities)]
            logger.info(f"Filtered by commodities {self.config.filter_commodities}: {len(df)} rows remaining")
            
        logger.info(f"Local data loaded and filtered from {initial_rows} to {len(df)} rows")
        return df

    def run_full_pipeline(
        self, 
        df_consolidated: pd.DataFrame = None
    ) -> Dict[str, Any]:
        """
        Run the complete feature engineering pipeline.
        
        Args:
            input_data: DataFrame or Path to consolidated data file.
                       If None, will use configuration-based data loading.
            
        Returns:
            Dictionary with pipeline results including both scaled and unscaled features
        """
        try:
            logger.info("Starting complete feature engineering pipeline")
            
            input_source = "unknown"
            
            if df_consolidated is not None:
                df_consolidated = self._filter_data(df_consolidated)
            
            # Load and filter from master data path 
            if df_consolidated is None:
                try:
                    df_consolidated = self._load_and_filter_data()
                    if df_consolidated is not None:
                        input_source = f"master_data_path: {self.config.master_data_path}"
                except FileNotFoundError as e:
                    logger.warning(f"{e}. Will try fallback method.")

            # Final validation: Ensure data was loaded
            if df_consolidated is None or df_consolidated.empty:
                raise ValueError("Failed to load any input data for the pipeline.")
            
            # Validate DataFrame has required columns
            required_columns = ['City', 'Commodity', 'Price', 'Date']
            missing_columns = [col for col in required_columns if col not in df_consolidated.columns]
            if missing_columns:
                raise ValueError(f"DataFrame missing required columns: {missing_columns}")
            
            logger.info(f"Data loaded successfully from: {input_source}")
            logger.info(f"Data shape: {df_consolidated.shape}")
            
            # Extract features
            unscaled_features = self.run_feature_extraction(df_consolidated)
            
            # Scale features
            scaled_results = self.run_feature_scaling(unscaled_features)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Prepare results
            results = {
                'success': True,
                'consolidated': df_consolidated,
                'unscaled_features': unscaled_features,
                'scaled_features': {
                    method: scaled_df for method, (scaled_df, _) in scaled_results.items()
                },
                'scalers': {
                    method: scaler for method, (_, scaler) in scaled_results.items()
                },
                # 'export_paths': export_results,
                'validation_results': self.extractor.validate_feature_matrix(unscaled_features),
                'input_source': input_source,
                'timestamp': timestamp,
                'aggregation_freq': self.config.aggregation_freq
            }
            
            logger.info("Feature engineering pipeline completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Feature engineering pipeline failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'unscaled_features': None,
                'scaled_features': None,
                'scalers': None,
                # 'export_paths': None,
                'validation_results': None,
                'input_source': None,
                'aggregation_freq': self.config.aggregation_freq if hasattr(self.config, 'aggregation_freq') else None
            }
