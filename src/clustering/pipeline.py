"""
Main Clustering Analysis Pipeline

This module orchestrates the complete clustering analysis pipeline,
supporting both batch processing and API integration.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Union, Optional, Tuple
import pandas as pd
import numpy as np

from .config import ClusteringPipelineConfig
from .input_handler import ClusteringInputHandler
from .algorithms import ClusteringAlgorithmManager, ClusteringResult
from .evaluator import ClusteringEvaluator
from .api_formatter import APIResponseFormatter
from src.utils.logger import setup_logging

logger = logging.getLogger(__name__)


class ClusteringAnalysisPipeline:
    """
    Main clustering analysis pipeline.
    
    Orchestrates:
    1. Data loading and validation
    2. Multiple clustering algorithms
    3. Comprehensive evaluation
    4. Visualization generation
    5. Results export (Excel, plots, API responses)
    
    Supports both file-based batch processing and DataFrame-based API calls.
    """
    
    def __init__(self, config: ClusteringPipelineConfig):
        """
        Initialize clustering analysis pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        
        # Setup logging
        if config.enable_file_logging:
            setup_logging(
                enable_file_logging=True,
                log_level=config.log_level,
                log_dir=config.log_dir
            )
        
        # Initialize components
        self.input_handler = ClusteringInputHandler(config.coordinates_path)
        self.algorithm_manager = ClusteringAlgorithmManager(config.random_state)
        self.evaluator = ClusteringEvaluator()
        self.api_formatter = APIResponseFormatter(self.input_handler.coordinates_data)
        
        # Initialize visualization pipeline (will be set later to avoid circular imports)
        self.visualization_pipeline = None
        
        # Validate configuration
        validation = config.validate()
        if validation['errors']:
            raise ValueError(f"Configuration errors: {validation['errors']}")
        
        if validation['warnings']:
            for warning in validation['warnings']:
                logger.warning(warning)
        
        logger.info(f"Initialized clustering pipeline with algorithms: {config.algorithms}")
    
    def set_visualization_pipeline(self, viz_pipeline):
        """Set visualization pipeline (to avoid circular imports)."""
        self.visualization_pipeline = viz_pipeline
    
    def run_complete_analysis(
        self,
        scaled_data: Optional[Union[pd.DataFrame, Path]] = None,
        preprocessed_data: Optional[Union[pd.DataFrame, Path]] = None
    ) -> Dict[str, Any]:
        """
        Run complete clustering analysis across all algorithms and k values.
        
        Args:
            scaled_data: Scaled feature data (DataFrame or path)
            preprocessed_data: Original preprocessed data for visualizations
            
        Returns:
            Dictionary with all clustering results
        """
        logger.info("Starting complete clustering analysis")
        start_time = time.time()
        
        # Use provided data or config paths
        scaled_input = scaled_data or self.config.scaled_data_path
        preprocessed_input = preprocessed_data or self.config.preprocessed_data_path
        
        if scaled_input is None:
            raise ValueError("No scaled data provided")
        
        # Prepare input data
        scaled_df, preprocessed_df = self.input_handler.prepare_input_data(
            scaled_input, preprocessed_input
        )
        
        # Extract feature matrix and city names
        X, cities = self.input_handler.prepare_feature_matrix(scaled_df)
        
        # Create results directory structure
        if self.config.save_outputs:
            self._create_directory_structure()
        
        # Run all algorithm-k combinations
        all_results = {}
        total_experiments = len(self.config.algorithms) * len(self.config.k_range)
        experiment_count = 0
        
        for algorithm in self.config.algorithms:
            algorithm_results = {}
            
            for k in self.config.k_range:
                experiment_count += 1
                logger.info(f"Running experiment {experiment_count}/{total_experiments}: {algorithm} k={k}")
                
                try:
                    # Run single clustering experiment
                    result = self._run_single_experiment(
                        X, cities, algorithm, k, scaled_df, preprocessed_df
                    )
                    algorithm_results[k] = result
                    
                except Exception as e:
                    logger.error(f"Experiment {algorithm} k={k} failed: {e}")
                    algorithm_results[k] = {"error": str(e)}
            
            all_results[algorithm] = algorithm_results
        
        # Generate comparison analysis
        if self.config.save_outputs:
            self._generate_comparison_analysis(all_results)
        
        total_time = time.time() - start_time
        logger.info(f"Complete analysis finished in {total_time:.2f} seconds")
        
        return {
            "results": all_results,
            "summary": {
                "total_experiments": total_experiments,
                "successful_experiments": sum(
                    1 for alg_results in all_results.values()
                    for result in alg_results.values()
                    if "error" not in result
                ),
                "total_time_seconds": total_time,
                "algorithms": self.config.algorithms,
                "k_range": list(self.config.k_range)
            }
        }
    
    def run_single_clustering(
        self,
        scaled_data: Union[pd.DataFrame, Path],
        algorithm: str,
        k: int,
        return_format: str = "full",
        preprocessed_data: Optional[Union[pd.DataFrame, Path]] = None
    ) -> Union[ClusteringResult, List[Dict], Dict]:
        """
        Run single clustering experiment with flexible output formats.
        
        Args:
            scaled_data: Input data (DataFrame or path)
            algorithm: Clustering algorithm name
            k: Number of clusters
            return_format: Output format ("full", "api", "simple", "detailed")
            preprocessed_data: Original data for visualizations (optional)
            
        Returns:
            Results in specified format:
            - "full": Complete ClusteringResult object
            - "api": List of dicts for API response
            - "simple": Basic dict with results
            - "detailed": Comprehensive dict with metrics
        """
        logger.info(f"Running single clustering: {algorithm} k={k}")
        
        # Prepare input data
        scaled_df, preprocessed_df = self.input_handler.prepare_input_data(
            scaled_data, preprocessed_data
        )
        
        # Extract feature matrix and city names
        X, cities = self.input_handler.prepare_feature_matrix(scaled_df)
        
        # Validate inputs
        validation = self.input_handler.validate_clustering_inputs(X, cities, k)
        if not validation['valid']:
            error_msg = f"Invalid inputs: {validation['errors']}"
            logger.error(error_msg)
            
            if return_format == "api":
                return self.api_formatter.format_error_response(error_msg, "ValidationError")
            else:
                raise ValueError(error_msg)
        
        try:
            # Run clustering
            result = self._fit_clustering_algorithm(X, cities, algorithm, k)
            
            # Generate outputs if not in API mode
            if self.config.save_outputs and not self.config.api_mode:
                output_dir = self.config.results_dir / algorithm / f"k{k}"
                self._generate_experiment_outputs(result, scaled_df, preprocessed_df, output_dir)
            
            # Format response based on return_format
            if return_format == "api":
                # Build frontend-friendly response including clusters palette and trends
                analysis_id = f"anl_{int(time.time())}"
                
                # Create merged_df with cluster labels for radar features
                labels = list(result.labels)
                cluster_map = pd.DataFrame({
                    "City": scaled_df["City"],
                    "Cluster": labels
                })
                merged_df = preprocessed_df.merge(cluster_map, on="City", how="left")
                
                # Generate PDF report
                pdf_path = self._generate_pdf_report(
                    scaled_df=scaled_df,
                    preprocessed_df=preprocessed_df,
                    labels=result.labels,
                    model=result.model,
                    silhouette_avg=result.additional_info.get("silhouette"),
                    algorithm_name=algorithm,
                    analysis_id=analysis_id
                )
                
                response = self.api_formatter.format_frontend_response(
                    analysis_id=analysis_id,
                    labels=result.labels,
                    cities=cities,
                    preprocessed_df=preprocessed_df,
                    merged_df=merged_df,
                    config=self.config,
                    scaled_features=scaled_df
                )
                
                # Add PDF info to response
                response["pdf_available"] = True
                response["pdf_path"] = f"temp_pdfs/analysis_{analysis_id}.pdf"
                
                return response
            elif return_format == "detailed":
                return self.api_formatter.format_detailed_response(
                    cities, result.labels, result.additional_info or {}, algorithm, k, result.execution_time
                )
            elif return_format == "simple":
                return self.api_formatter.format_simple_response(cities, result.labels, algorithm, k)
            else:  # "full"
                return result
                
        except Exception as e:
            error_msg = f"Clustering failed: {str(e)}"
            logger.error(error_msg)
            
            if return_format in ["api", "detailed"]:
                return self.api_formatter.format_error_response(error_msg, "ClusteringError")
            else:
                raise
    
    def _run_single_experiment(
        self,
        X: np.ndarray,
        cities: List[str],
        algorithm: str,
        k: int,
        scaled_df: pd.DataFrame,
        preprocessed_df: Optional[pd.DataFrame]
    ) -> Dict[str, Any]:
        """Run a single clustering experiment."""
        try:
            # Fit clustering algorithm
            result = self._fit_clustering_algorithm(X, cities, algorithm, k)
            
            # Generate outputs
            if self.config.save_outputs:
                output_dir = self.config.results_dir / algorithm / f"k{k}"
                self._generate_experiment_outputs(result, scaled_df, preprocessed_df, output_dir)
            
            # Return summary
            return {
                "algorithm": algorithm,
                "k": k,
                "labels": result.labels.tolist(),
                "metrics": result.additional_info or {},
                "execution_time": result.execution_time,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Experiment {algorithm} k={k} failed: {e}")
            return {
                "algorithm": algorithm,
                "k": k,
                "error": str(e),
                "success": False
            }
    
    def _fit_clustering_algorithm(
        self,
        X: np.ndarray,
        cities: List[str],
        algorithm: str,
        k: int
    ) -> ClusteringResult:
        """Fit clustering algorithm and evaluate results."""
        # Get algorithm parameters
        algorithm_params = self.config.get_algorithm_params(algorithm)
        
        # Validate parameters
        param_validation = self.algorithm_manager.validate_algorithm_params(algorithm, algorithm_params)
        if not param_validation['valid']:
            raise ValueError(f"Invalid parameters for {algorithm}: {param_validation['errors']}")
        
        # Fit clustering algorithm
        result = self.algorithm_manager.fit_clustering(X, algorithm, k, algorithm_params)
        
        # Evaluate clustering
        metrics = self.evaluator.evaluate_clustering(X, result.labels, algorithm, result.centers)
        
        # Add metrics to result
        if result.additional_info is None:
            result.additional_info = {}
        result.additional_info.update(metrics)
        
        logger.info(f"Clustering completed: {algorithm} k={k}, silhouette={metrics.get('silhouette', 'N/A'):.3f}")
        
        return result

    def _generate_pdf_report(
        self,
        scaled_df: pd.DataFrame,
        preprocessed_df: pd.DataFrame,
        labels: np.ndarray,
        model: Any,
        silhouette_avg: float,
        algorithm_name: str,
        analysis_id: str
    ) -> Path:
        """
        Generate PDF report using visualization service.
        """
        try:
            # Create temp_pdfs directory
            temp_pdfs_dir = Path("temp_pdfs")
            temp_pdfs_dir.mkdir(exist_ok=True)
            
            # PDF output path
            pdf_path = temp_pdfs_dir / f"analysis_{analysis_id}.pdf"
            
            # Import visualization service
            from src.visualization.pipeline import ClusterVisualizationService
            
            # Initialize visualization service
            viz_service = ClusterVisualizationService(
                scaled_df=scaled_df,
                preprocessed_df=preprocessed_df,
                labels=labels,
                model=model,
                silhouette_avg=silhouette_avg,
                algorithm_name=algorithm_name,
                output_path=temp_pdfs_dir 
            )
            
            # Generate PDF report
            viz_service.generate_pdf_report(pdf_path)
            
            logger.info(f"PDF report generated: {pdf_path}")
            return pdf_path
            
        except Exception as e:
            logger.error(f"Error generating PDF report: {e}")
            # Return None to indicate PDF generation failed
            return None

    def _generate_experiment_outputs(
        self,
        result: ClusteringResult,
        scaled_df: pd.DataFrame,
        preprocessed_df: Optional[pd.DataFrame],
        output_dir: Path
    ):
        """Generate all outputs for a single experiment."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create visualizations directory
        viz_dir = output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # Generate visualizations if pipeline is available
        if self.visualization_pipeline is not None:
            try:
                self.visualization_pipeline.generate_all_visualizations(
                    result, scaled_df, preprocessed_df, viz_dir
                )
            except Exception as e:
                logger.error(f"Visualization generation failed: {e}")
        
        # Save metrics
        if self.config.save_metrics:
            metrics_file = output_dir / "metrics.json"
            self._save_metrics(result, metrics_file)
        
        # Generate Excel reports if enabled
        if self.config.save_excel and hasattr(self, 'excel_generator'):
            try:
                self.excel_generator.generate_cluster_reports(result, scaled_df, output_dir)
            except Exception as e:
                logger.error(f"Excel generation failed: {e}")
    
    def _create_directory_structure(self):
        """Create directory structure for results."""
        base_dir = self.config.results_dir
        base_dir.mkdir(parents=True, exist_ok=True)
        
        for algorithm in self.config.algorithms:
            alg_dir = base_dir / algorithm
            alg_dir.mkdir(exist_ok=True)
            
            for k in self.config.k_range:
                k_dir = alg_dir / f"k{k}"
                k_dir.mkdir(exist_ok=True)
                
                # Create subdirectories
                (k_dir / "visualizations").mkdir(exist_ok=True)
        
        logger.info(f"Created directory structure in {base_dir}")
    
    def _save_metrics(self, result: ClusteringResult, output_file: Path):
        """Save clustering metrics to JSON file."""
        import json
        
        metrics_data = {
            "algorithm": result.algorithm,
            "k": result.k,
            "execution_time": result.execution_time,
            "metrics": result.additional_info or {},
            "cluster_sizes": pd.Series(result.labels).value_counts().sort_index().to_dict()
        }
        
        with open(output_file, 'w') as f:
            json.dump(metrics_data, f, indent=2, default=str)
        
        logger.debug(f"Saved metrics to {output_file}")
    
    def _generate_comparison_analysis(self, all_results: Dict[str, Dict[int, Any]]):
        """Generate cross-algorithm comparison analysis."""
        try:
            # Extract successful results for comparison
            comparison_data = []
            
            for algorithm, alg_results in all_results.items():
                for k, result in alg_results.items():
                    if "error" not in result and "metrics" in result:
                        comparison_data.append({
                            "algorithm": algorithm,
                            "k": k,
                            **result["metrics"]
                        })
            
            if comparison_data:
                # Save comparison data
                comparison_df = pd.DataFrame(comparison_data)
                comparison_file = self.config.results_dir / "comparison_summary.csv"
                comparison_df.to_csv(comparison_file, index=False)
                
                logger.info(f"Saved comparison analysis to {comparison_file}")
            
        except Exception as e:
            logger.error(f"Failed to generate comparison analysis: {e}")
    
    def get_best_clustering(
        self,
        results: Dict[str, Dict[int, Any]],
        metric: str = "silhouette"
    ) -> Optional[Dict[str, Any]]:
        """
        Identify best clustering result based on specified metric.
        
        Args:
            results: All clustering results
            metric: Metric to optimize ("silhouette", "davies_bouldin", "calinski_harabasz")
            
        Returns:
            Best clustering result or None if no valid results
        """
        best_result = None
        best_score = float('-inf') if metric in ["silhouette", "calinski_harabasz"] else float('inf')
        
        for algorithm, alg_results in results.items():
            for k, result in alg_results.items():
                if "error" not in result and "metrics" in result:
                    score = result["metrics"].get(metric)
                    
                    if score is not None:
                        is_better = (
                            (metric in ["silhouette", "calinski_harabasz"] and score > best_score) or
                            (metric == "davies_bouldin" and score < best_score)
                        )
                        
                        if is_better:
                            best_score = score
                            best_result = {
                                "algorithm": algorithm,
                                "k": k,
                                "score": score,
                                "result": result
                            }
        
        return best_result
