"""
Input Handler for Clustering Pipeline

This module handles flexible input types for the clustering pipeline,
supporting both file paths and pandas DataFrames.
"""

import logging
from pathlib import Path
from typing import Union, Optional, Tuple, Dict, Any, List
import pandas as pd
import numpy as np
import json

logger = logging.getLogger(__name__)


class ClusteringInputHandler:
    """
    Handles multiple input types for clustering pipeline.
    
    Supports:
    - File paths (CSV, Excel)
    - pandas DataFrames
    - Input validation and enrichment
    """
    
    def __init__(self, coordinates_path: Path):
        """
        Initialize input handler.
        
        Args:
            coordinates_path: Path to city coordinates JSON file
        """
        self.coordinates_path = coordinates_path
        self.coordinates_data = self._load_coordinates()
        
    def _ensure_city_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure the DataFrame has a 'City' column.
        Accepts either an explicit 'City' column or an index named 'City' (or unnamed index of city names).
        """
        if 'City' in df.columns:
            return df
        # If index is named 'City', reset to column
        if df.index.name == 'City':
            return df.reset_index()
        # If index is unnamed, assume it holds city names and move to 'City'
        if df.index.name is None:
            df = df.reset_index()
            if 'City' not in df.columns:
                df = df.rename(columns={'index': 'City'})
            return df
        raise ValueError("City identifier not found. Provide 'City' column or set index name to 'City'.")

    def _load_coordinates(self) -> pd.DataFrame:
        """Load city coordinates from JSON file."""
        try:
            with open(self.coordinates_path, 'r') as f:
                coordinates_dict = json.load(f)
            
            # Convert to DataFrame
            coords_list = []
            for city, coords in coordinates_dict.items():
                coords_list.append({
                    'City': city,
                    'Latitude': coords[0],
                    'Longitude': coords[1]
                })
            
            df = pd.DataFrame(coords_list)
            logger.info(f"Loaded coordinates for {len(df)} cities")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load coordinates from {self.coordinates_path}: {e}")
            return pd.DataFrame(columns=['City', 'Latitude', 'Longitude'])
    
    def prepare_input_data(
        self,
        scaled_data: Union[pd.DataFrame, Path, str],
        preprocessed_data: Optional[Union[pd.DataFrame, Path, str]] = None
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Prepare input data from various sources.
        
        Args:
            scaled_data: DataFrame or path to scaled feature matrix
            preprocessed_data: DataFrame or path to preprocessed data (optional)
            
        Returns:
            Tuple of (scaled_df, preprocessed_df)
            
        Raises:
            ValueError: If input data is invalid
            FileNotFoundError: If file paths don't exist
        """
        logger.info("Preparing input data for clustering")
        
        # Load scaled data
        if isinstance(scaled_data, pd.DataFrame):
            scaled_df = scaled_data.copy()
            logger.info(f"Using provided DataFrame with {len(scaled_df)} rows")
        else:
            scaled_df = self._load_dataframe(scaled_data, "scaled features")
        # Normalize City column
        scaled_df = self._ensure_city_column(scaled_df)
        
        # Validate scaled data
        validation_result = self.validate_scaled_dataframe(scaled_df)
        if not validation_result['valid']:
            raise ValueError(f"Invalid scaled data: {validation_result['errors']}")
        
        if validation_result['warnings']:
            for warning in validation_result['warnings']:
                logger.warning(warning)
        
        # Load preprocessed data (optional)
        preprocessed_df = None
        if preprocessed_data is not None:
            if isinstance(preprocessed_data, pd.DataFrame):
                preprocessed_df = preprocessed_data.copy()
                logger.info(f"Using provided preprocessed DataFrame with {len(preprocessed_df)} rows")
            else:
                preprocessed_df = self._load_dataframe(preprocessed_data, "preprocessed data")
            # Normalize City column for preprocessed
            preprocessed_df = self._ensure_city_column(preprocessed_df)
        
        logger.info(f"Successfully prepared input data: {len(scaled_df)} cities, {len([c for c in scaled_df.columns if c != 'City'])} features")
        
        return scaled_df, preprocessed_df
    
    def _load_dataframe(self, file_path: Union[Path, str], data_type: str) -> pd.DataFrame:
        """
        Load DataFrame from file path.
        
        Args:
            file_path: Path to data file
            data_type: Description of data type for logging
            
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is unsupported
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"{data_type} file not found: {file_path}")
        
        logger.info(f"Loading {data_type} from {file_path}")
        
        try:
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            logger.info(f"Successfully loaded {data_type}: {len(df)} rows, {len(df.columns)} columns")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load {data_type} from {file_path}: {e}")
            raise
    
    def validate_scaled_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate that DataFrame has required structure for clustering.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with validation results:
            {
                'valid': bool,
                'errors': List[str],
                'warnings': List[str],
                'feature_count': int,
                'city_count': int
            }
        """
        # Normalize to ensure 'City' column exists
        df = self._ensure_city_column(df)
        errors = []
        warnings = []
        
        # Check basic structure
        if df.empty:
            errors.append("DataFrame is empty")
            return {
                'valid': False,
                'errors': errors,
                'warnings': warnings,
                'feature_count': 0,
                'city_count': 0
            }
        
        # Required columns
        if 'City' not in df.columns:
            errors.append("Missing required 'City' column")
        
        # Feature columns (should be numeric)
        feature_cols = [col for col in df.columns if col != 'City']
        if len(feature_cols) == 0:
            errors.append("No feature columns found")
        
        # Check for numeric features
        non_numeric = []
        for col in feature_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                non_numeric.append(col)
        
        if non_numeric:
            errors.append(f"Non-numeric feature columns: {non_numeric}")
        
        # Check for missing values
        missing_data = df.isnull().sum()
        if missing_data.any():
            missing_info = missing_data[missing_data > 0].to_dict()
            warnings.append(f"Missing values detected: {missing_info}")
        
        # Check for infinite values
        if df.select_dtypes(include=[np.number]).isin([np.inf, -np.inf]).any().any():
            warnings.append("Infinite values detected in numeric columns")
        
        # Check city names against coordinates
        if not self.coordinates_data.empty and 'City' in df.columns:
            cities_without_coords = set(df['City']) - set(self.coordinates_data['City'])
            if cities_without_coords:
                warnings.append(f"Cities without coordinates: {sorted(list(cities_without_coords))}")
        
        # Check for duplicate cities
        if 'City' in df.columns:
            duplicate_cities = df['City'].duplicated().sum()
            if duplicate_cities > 0:
                errors.append(f"Duplicate cities found: {duplicate_cities} duplicates")
        
        # Check minimum number of cities for clustering
        city_count = len(df) if 'City' in df.columns else 0
        if city_count < 2:
            errors.append("At least 2 cities required for clustering")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'feature_count': len(feature_cols),
            'city_count': city_count
        }
    
    def enrich_with_coordinates(self, cities: List[str]) -> pd.DataFrame:
        """
        Add coordinate information to city list.
        
        Args:
            cities: List of city names
            
        Returns:
            DataFrame with City, Latitude, Longitude columns
        """
        city_df = pd.DataFrame({'City': cities})
        
        if self.coordinates_data.empty:
            logger.warning("No coordinates data available")
            city_df['Latitude'] = None
            city_df['Longitude'] = None
            return city_df
        
        # Merge with coordinates
        enriched_df = city_df.merge(
            self.coordinates_data,
            on='City',
            how='left'
        )
        
        # Log missing coordinates
        missing_coords = enriched_df['Latitude'].isnull().sum()
        if missing_coords > 0:
            logger.warning(f"{missing_coords} cities missing coordinate information")
        
        return enriched_df
    
    def prepare_feature_matrix(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Extract feature matrix and city names from DataFrame.
        
        Args:
            df: DataFrame with 'City' column and feature columns
            
        Returns:
            Tuple of (feature_matrix, city_names)
        """
        # Normalize to ensure 'City' column exists
        df = self._ensure_city_column(df)
        if 'City' not in df.columns:
            raise ValueError("DataFrame must contain 'City' column")
        
        # Extract feature columns (all except 'City')
        feature_cols = [col for col in df.columns if col != 'City']
        
        if len(feature_cols) == 0:
            raise ValueError("No feature columns found")
        
        # Extract feature matrix
        X = df[feature_cols].values
        
        # Extract city names
        cities = df['City'].tolist()
        
        logger.info(f"Prepared feature matrix: {X.shape[0]} cities × {X.shape[1]} features")
        
        return X, cities
    
    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of feature column names.
        
        Args:
            df: DataFrame with features
            
        Returns:
            List of feature column names
        """
        return [col for col in df.columns if col != 'City']
    
    def validate_clustering_inputs(
        self,
        X: np.ndarray,
        cities: List[str],
        k: int
    ) -> Dict[str, Any]:
        """
        Validate inputs for clustering algorithm.
        
        Args:
            X: Feature matrix
            cities: List of city names
            k: Number of clusters
            
        Returns:
            Validation results dictionary
        """
        errors = []
        warnings = []
        
        # Check feature matrix
        if X.size == 0:
            errors.append("Feature matrix is empty")
        elif len(X.shape) != 2:
            errors.append(f"Feature matrix must be 2D, got shape {X.shape}")
        
        # Check for NaN or infinite values
        if np.isnan(X).any():
            errors.append("Feature matrix contains NaN values")
        if np.isinf(X).any():
            errors.append("Feature matrix contains infinite values")
        
        # Check cities list
        if len(cities) != X.shape[0]:
            errors.append(f"Number of cities ({len(cities)}) doesn't match feature matrix rows ({X.shape[0]})")
        
        # Check k value
        if k < 2:
            errors.append("Number of clusters (k) must be at least 2")
        elif k >= len(cities):
            errors.append(f"Number of clusters (k={k}) must be less than number of cities ({len(cities)})")
        
        # Check feature variance
        if X.shape[1] > 0:
            feature_vars = np.var(X, axis=0)
            zero_var_features = np.sum(feature_vars == 0)
            if zero_var_features > 0:
                warnings.append(f"{zero_var_features} features have zero variance")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
