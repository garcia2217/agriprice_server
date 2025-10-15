"""
Feature Scaling Module

This module handles the scaling of feature matrices using different scaling methods
(StandardScaler, MinMaxScaler, RobustScaler) for clustering analysis.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
from datetime import datetime
from pathlib import Path
import json

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from .config import FeatureEngineeringConfig

logger = logging.getLogger(__name__)


class FeatureScaler:
    """
    Handles scaling of feature matrices using multiple scaling methods.
    
    Supports StandardScaler, MinMaxScaler, and RobustScaler from scikit-learn.
    Returns both scaled and unscaled feature matrices.
    """
    
    def __init__(self, config: FeatureEngineeringConfig):
        """
        Initialize the feature scaler.
        
        Args:
            config: Configuration object for feature engineering
        """
        self.config = config
        self.scalers = {}
        
    def _get_scaler(self, method: str):
        """
        Get scaler instance for the specified method.
        
        Args:
            method: Scaling method ('standard', 'minmax', 'robust')
            
        Returns:
            Scaler instance
        """
        if method == 'standard':
            return StandardScaler()
        elif method == 'minmax':
            return MinMaxScaler()
        elif method == 'robust':
            return RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
    
    def scale_features(
        self, 
        feature_matrix: pd.DataFrame, 
        method: str
    ) -> Tuple[pd.DataFrame, Any]:
        """
        Scale feature matrix using the specified method.
        
        Args:
            feature_matrix: Original unscaled feature DataFrame
            method: Scaling method ('standard', 'minmax', 'robust')
            
        Returns:
            Tuple of (scaled_dataframe, fitted_scaler)
        """
        logger.info(f"Scaling features using {method} method")
        
        # Select only numeric feature columns
        numeric_cols = feature_matrix.select_dtypes(include=[np.number]).columns.tolist()
        X = feature_matrix[numeric_cols].values
        
        # Get and fit scaler
        scaler = self._get_scaler(method)
        X_scaled = scaler.fit_transform(X)
        
        # Build scaled DataFrame (preserve index and columns)
        scaled_df = pd.DataFrame(
            X_scaled, 
            index=feature_matrix.index, 
            columns=numeric_cols
        )
        
        # Store scaler for later use
        self.scalers[method] = scaler
        
        # Log scaling statistics
        logger.info(
            f"  {method} scaled: mean={scaled_df.values.mean():.4f}, "
            f"std={scaled_df.values.std():.4f}"
        )
        
        return scaled_df, scaler
    
    def scale_all_methods(
        self, 
        feature_matrix: pd.DataFrame
    ) -> Dict[str, Tuple[pd.DataFrame, Any]]:
        """
        Scale feature matrix using all configured methods.
        
        Args:
            feature_matrix: Original unscaled feature DataFrame
            
        Returns:
            Dictionary mapping method names to (scaled_dataframe, scaler) tuples
        """
        logger.info(f"Scaling features using {len(self.config.scaling_methods)} methods")
        
        scaled_results = {}
        
        for method in self.config.scaling_methods:
            scaled_df, scaler = self.scale_features(feature_matrix, method)
            scaled_results[method] = (scaled_df, scaler)
        
        logger.info("All scaling methods completed")
        return scaled_results
    
    def export_scaled_features(
        self, 
        scaled_results: Dict[str, Tuple[pd.DataFrame, Any]], 
        output_dir: Path,
        timestamp: str = None
    ) -> Dict[str, Dict[str, str]]:
        """
        Export scaled feature matrices to files.
        
        Args:
            scaled_results: Dictionary of scaling results
            output_dir: Directory to save outputs
            timestamp: Optional timestamp for filenames
            
        Returns:
            Dictionary mapping method names to export paths
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        output_dir = output_dir / "scaled"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        all_export_paths = {}
        
        for method, (scaled_df, scaler) in scaled_results.items():
            logger.info(f"Exporting {method} scaled features")
            
            # Generate filenames
            base_filename = f"feature_matrix_{method}_scaled_{timestamp}"
            export_paths = {}
            
            # Export in configured formats
            for export_format in self.config.export_formats:
                if export_format == "csv":
                    csv_path = output_dir / f"{base_filename}.csv"
                    scaled_df.to_csv(csv_path)
                    export_paths['csv'] = str(csv_path)
                    
                elif export_format == "excel":
                    excel_path = output_dir / f"{base_filename}.xlsx"
                    scaled_df.to_excel(excel_path)
                    export_paths['excel'] = str(excel_path)
                    
                elif export_format == "json":
                    json_path = output_dir / f"{base_filename}.json"
                    scaled_df.to_json(json_path, orient='index', indent=2)
                    export_paths['json'] = str(json_path)
            
            # Export metadata
            metadata = {
                'export_timestamp': datetime.now().isoformat(),
                'scaling_method': method,
                'shape': scaled_df.shape,
                'cities': scaled_df.index.tolist(),
                'columns': scaled_df.columns.tolist(),
                'scaler_params': scaler.get_params() if hasattr(scaler, 'get_params') else {},
                'statistics': {
                    'mean': scaled_df.values.mean(),
                    'std': scaled_df.values.std(),
                    'min': scaled_df.values.min(),
                    'max': scaled_df.values.max()
                }
            }
            
            metadata_path = output_dir / f"{base_filename}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            export_paths['metadata'] = str(metadata_path)
            
            all_export_paths[method] = export_paths
            
            logger.info(f"  Saved: {base_filename}.csv, {base_filename}.xlsx")
        
        return all_export_paths
    
    def export_unscaled_features(
        self, 
        feature_matrix: pd.DataFrame, 
        output_dir: Path,
        timestamp: str = None
    ) -> Dict[str, str]:
        """
        Export unscaled feature matrix to files.
        
        Args:
            feature_matrix: Original unscaled feature DataFrame
            output_dir: Directory to save outputs
            timestamp: Optional timestamp for filenames
            
        Returns:
            Dictionary with export file paths
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        base_filename = f"feature_matrix_unscaled_{timestamp}"
        export_paths = {}
        
        logger.info("Exporting unscaled feature matrix")
        
        # Export in configured formats
        for export_format in self.config.export_formats:
            if export_format == "csv":
                csv_path = output_dir / f"{base_filename}.csv"
                feature_matrix.to_csv(csv_path)
                export_paths['csv'] = str(csv_path)
                
            elif export_format == "excel":
                excel_path = output_dir / f"{base_filename}.xlsx"
                feature_matrix.to_excel(excel_path)
                export_paths['excel'] = str(excel_path)
                
            elif export_format == "json":
                json_path = output_dir / f"{base_filename}.json"
                feature_matrix.to_json(json_path, orient='index', indent=2)
                export_paths['json'] = str(json_path)
        
        # Export metadata
        numeric_features = feature_matrix.select_dtypes(include=[np.number]).columns.tolist()
        metadata = {
            'export_timestamp': datetime.now().isoformat(),
            'scaling_method': 'none',
            'shape': feature_matrix.shape,
            'cities': feature_matrix.index.tolist(),
            'columns': feature_matrix.columns.tolist(),
            'numeric_features': numeric_features,
            'statistics': {
                'mean': feature_matrix[numeric_features].values.mean(),
                'std': feature_matrix[numeric_features].values.std(),
                'min': feature_matrix[numeric_features].values.min(),
                'max': feature_matrix[numeric_features].values.max()
            }
        }
        
        metadata_path = output_dir / f"{base_filename}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        export_paths['metadata'] = str(metadata_path)
        
        logger.info(f"  Saved: {base_filename}.csv, {base_filename}.xlsx")
        
        return export_paths
