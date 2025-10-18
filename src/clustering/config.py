"""
Clustering Pipeline Configuration

This module provides configuration classes for the clustering analysis pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, Literal
# import yaml
import json


@dataclass
class ClusteringPipelineConfig:
    """
    Configuration for clustering analysis pipeline.
    
    Supports both file-based batch processing and DataFrame-based API calls.
    """
    
    # Data paths (optional for API mode)
    scaled_data_path: Optional[Path] = None
    preprocessed_data_path: Optional[Path] = None
    coordinates_path: Path = Path("data/city_coordinates.json")
    
    # Clustering parameters
    algorithms: List[str] = field(default_factory=lambda: ["kmeans", "fcm", "spectral"])
    k_range: range = field(default_factory=lambda: range(2, 11))
    random_state: int = 42
    
    # Algorithm-specific parameters
    kmeans_params: Dict[str, Any] = field(default_factory=lambda: {
        "init": "k-means++",
        "n_init": 10,
        "max_iter": 300,
        "tol": 1e-4
    })
    
    fcm_params: Dict[str, Any] = field(default_factory=lambda: {
        "m": 2.0,  # Fuzziness parameter
        "max_iter": 1000,
        "error": 1e-5
    })
    
    spectral_params: Dict[str, Any] = field(default_factory=lambda: {
        "affinity": "rbf",
        "gamma": 1.0,
        "n_init": 10
    })
    
    # Output configuration
    results_dir: Path = Path("clustering_results")
    save_outputs: bool = True
    save_plots: bool = True
    save_excel: bool = True
    save_metrics: bool = True
    
    # API configuration
    api_mode: bool = False
    return_format: str = "full"  # "full", "api", "simple"
    
    # Correlation analysis configuration
    correlation_mode: Literal["global", "per_cluster"] = "global"
    
    # Visualization configuration
    figure_size: tuple = (16, 10)
    dpi: int = 300
    color_palette: str = "tab10"
    save_format: str = "png"
    
    # Logging configuration
    log_level: str = "INFO"
    log_dir: Path = Path("logs")
    enable_file_logging: bool = True
    
    def __post_init__(self):
        """Convert string paths to Path objects."""
        if isinstance(self.scaled_data_path, str):
            self.scaled_data_path = Path(self.scaled_data_path)
        if isinstance(self.preprocessed_data_path, str):
            self.preprocessed_data_path = Path(self.preprocessed_data_path)
        if isinstance(self.coordinates_path, str):
            self.coordinates_path = Path(self.coordinates_path)
        if isinstance(self.results_dir, str):
            self.results_dir = Path(self.results_dir)
        if isinstance(self.log_dir, str):
            self.log_dir = Path(self.log_dir)
    
    # @classmethod
    # def from_yaml(cls, config_path: Union[str, Path]) -> 'ClusteringPipelineConfig':
    #     """
    #     Load configuration from YAML file.
        
    #     Args:
    #         config_path: Path to YAML configuration file
            
    #     Returns:
    #         ClusteringPipelineConfig instance
    #     """
    #     config_path = Path(config_path)
        
    #     if not config_path.exists():
    #         raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    #     with open(config_path, 'r') as f:
    #         config_dict = yaml.safe_load(f)
        
    #     # Flatten nested configuration
    #     flat_config = {}
        
    #     # Handle data section
    #     if 'data' in config_dict:
    #         data_config = config_dict['data']
    #         flat_config.update({
    #             'scaled_data_path': data_config.get('scaled_features_path'),
    #             'preprocessed_data_path': data_config.get('preprocessed_data_path'),
    #             'coordinates_path': data_config.get('coordinates_path', 'data/city_coordinates.json')
    #         })
        
    #     # Handle clustering section
    #     if 'clustering' in config_dict:
    #         clustering_config = config_dict['clustering']
    #         flat_config.update({
    #             'algorithms': clustering_config.get('algorithms', ['kmeans', 'fcm', 'spectral']),
    #             'random_state': clustering_config.get('random_state', 42)
    #         })
            
    #         # Handle k_range
    #         k_range_config = clustering_config.get('k_range', [2, 10])
    #         if isinstance(k_range_config, list) and len(k_range_config) == 2:
    #             flat_config['k_range'] = range(k_range_config[0], k_range_config[1] + 1)
            
    #         # Algorithm-specific parameters
    #         flat_config['kmeans_params'] = clustering_config.get('kmeans', {})
    #         flat_config['fcm_params'] = clustering_config.get('fcm', {})
    #         flat_config['spectral_params'] = clustering_config.get('spectral', {})
        
    #     # Handle output section
    #     if 'output' in config_dict:
    #         output_config = config_dict['output']
    #         flat_config.update({
    #             'results_dir': output_config.get('results_dir', 'clustering_results'),
    #             'save_plots': output_config.get('save_plots', True),
    #             'save_excel': output_config.get('save_excel', True),
    #             'save_metrics': output_config.get('save_metrics', True)
    #         })
        
    #     # Handle visualization section
    #     if 'visualization' in config_dict:
    #         viz_config = config_dict['visualization']
    #         flat_config.update({
    #             'figure_size': tuple(viz_config.get('figure_size', [16, 10])),
    #             'dpi': viz_config.get('dpi', 300),
    #             'color_palette': viz_config.get('color_palette', 'tab10'),
    #             'save_format': viz_config.get('save_format', 'png')
    #         })
        
    #     # Handle logging section
    #     if 'logging' in config_dict:
    #         log_config = config_dict['logging']
    #         flat_config.update({
    #             'log_level': log_config.get('level', 'INFO'),
    #             'log_dir': log_config.get('log_dir', 'logs'),
    #             'enable_file_logging': log_config.get('enable_file_logging', True)
    #         })
        
    #     return cls(**flat_config)
    
    @classmethod
    def for_api_call(
        cls,
        coordinates_path: Union[str, Path],
        algorithm: str = "kmeans",
        k: int = 3
    ) -> 'ClusteringPipelineConfig':
        """
        Create configuration optimized for API calls.
        
        Args:
            coordinates_path: Path to city coordinates file
            algorithm: Clustering algorithm to use
            k: Number of clusters
            
        Returns:
            ClusteringPipelineConfig instance for API use
        """
        return cls(
            coordinates_path=Path(coordinates_path),
            algorithms=[algorithm],
            k_range=range(k, k + 1),
            save_outputs=False,
            save_plots=False,
            save_excel=False,
            save_metrics=False,
            api_mode=True,
            return_format="api",
            enable_file_logging=False
        )
    
    def get_algorithm_params(self, algorithm: str) -> Dict[str, Any]:
        """
        Get parameters for specific algorithm.
        
        Args:
            algorithm: Algorithm name
            
        Returns:
            Dictionary of algorithm parameters
        """
        param_map = {
            'kmeans': self.kmeans_params,
            'fcm': self.fcm_params,
            'spectral': self.spectral_params
        }
        
        return param_map.get(algorithm, {})
    
    def validate(self) -> Dict[str, List[str]]:
        """
        Validate configuration settings.
        
        Returns:
            Dictionary with 'errors' and 'warnings' lists
        """
        errors = []
        warnings = []
        
        # Check required paths for non-API mode
        if not self.api_mode:
            if self.scaled_data_path and not self.scaled_data_path.exists():
                errors.append(f"Scaled data path does not exist: {self.scaled_data_path}")
            
            if self.preprocessed_data_path and not self.preprocessed_data_path.exists():
                warnings.append(f"Preprocessed data path does not exist: {self.preprocessed_data_path}")
        
        # Check coordinates file
        if not self.coordinates_path.exists():
            errors.append(f"Coordinates file does not exist: {self.coordinates_path}")
        
        # Validate algorithms
        valid_algorithms = {'kmeans', 'fcm', 'spectral'}
        invalid_algorithms = set(self.algorithms) - valid_algorithms
        if invalid_algorithms:
            errors.append(f"Invalid algorithms: {invalid_algorithms}")
        
        # Validate k_range
        if len(self.k_range) == 0:
            errors.append("k_range cannot be empty")
        elif min(self.k_range) < 2:
            errors.append("Minimum k value must be at least 2")
        
        return {'errors': errors, 'warnings': warnings}


@dataclass
class VisualizationConfig:
    """Configuration for visualization components."""
    
    figure_size: tuple = (16, 10)
    dpi: int = 300
    color_palette: str = "tab10"
    save_format: str = "png"
    
    # Plot-specific settings
    silhouette_plot_size: tuple = (10, 7)
    scatter_plot_size: tuple = (12, 8)
    trend_plot_size: tuple = (17, 12)
    
    # Styling
    grid_alpha: float = 0.3
    line_width: float = 2.0
    marker_size: float = 70
    legend_fontsize: int = 11
    title_fontsize: int = 15
