"""
Feature Engineering Module

This module provides modular components for extracting features from consolidated
time series food price data and preparing them for clustering analysis.
"""

from .config import FeatureEngineeringConfig
from .extractor import FeatureExtractor
from .scaler import FeatureScaler
from .pipeline import FeatureEngineeringPipeline

__all__ = [
    'FeatureEngineeringConfig',
    'FeatureExtractor', 
    'FeatureScaler',
    'FeatureEngineeringPipeline'
]
