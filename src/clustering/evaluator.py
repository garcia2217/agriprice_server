"""
Clustering Evaluation Module

This module provides comprehensive evaluation metrics for clustering results.
"""

import logging
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
from sklearn.metrics import (
    silhouette_score, 
    davies_bouldin_score, 
    calinski_harabasz_score,
    silhouette_samples
)
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


class ClusteringEvaluator:
    """
    Comprehensive clustering evaluation using multiple metrics.
    
    Provides:
    - Internal validation metrics (silhouette, Davies-Bouldin, Calinski-Harabasz)
    - Cluster quality analysis
    - Stability assessment
    - Custom metrics for food price clustering
    """
    
    def __init__(self):
        """Initialize clustering evaluator."""
        logger.info("Initialized clustering evaluator")
    
    def evaluate_clustering(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        algorithm: str = "unknown",
        centers: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Comprehensive clustering evaluation.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            labels: Cluster labels
            algorithm: Algorithm name for context
            centers: Cluster centers (if available)
            
        Returns:
            Dictionary of evaluation metrics
        """
        if len(np.unique(labels)) < 2:
            logger.warning("Cannot evaluate clustering with less than 2 clusters")
            return {"error": "insufficient_clusters"}
        
        logger.info(f"Evaluating {algorithm} clustering with {len(np.unique(labels))} clusters")
        
        metrics = {}
        
        try:
            # Core clustering metrics
            metrics.update(self._calculate_core_metrics(X, labels))
            
            # Cluster quality metrics
            metrics.update(self._calculate_quality_metrics(X, labels, centers))
            
            # Algorithm-specific metrics
            if algorithm == "kmeans" and centers is not None:
                metrics.update(self._calculate_kmeans_metrics(X, labels, centers))
            
            logger.info(f"Evaluation completed: Silhouette={metrics.get('silhouette', 'N/A'):.3f}")
            
        except Exception as e:
            logger.error(f"Error during clustering evaluation: {e}")
            metrics["evaluation_error"] = str(e)
        
        return metrics
    
    def _calculate_core_metrics(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Calculate core clustering validation metrics."""
        metrics = {}
        
        try:
            # Silhouette Score (higher is better, range: -1 to 1)
            silhouette = silhouette_score(X, labels)
            metrics["silhouette"] = float(silhouette)
            
            # Davies-Bouldin Index (lower is better, range: 0 to inf)
            db_score = davies_bouldin_score(X, labels)
            metrics["davies_bouldin"] = float(db_score)
            
            # Calinski-Harabasz Index (higher is better, range: 0 to inf)
            ch_score = calinski_harabasz_score(X, labels)
            metrics["calinski_harabasz"] = float(ch_score)
            
        except Exception as e:
            logger.warning(f"Error calculating core metrics: {e}")
        
        return metrics
    
    def _calculate_quality_metrics(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        centers: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate cluster quality and separation metrics."""
        metrics = {}
        
        try:
            # Cluster sizes
            unique_labels, counts = np.unique(labels, return_counts=True)
            cluster_sizes = dict(zip(unique_labels, counts))
            
            # Size-based metrics
            metrics["min_cluster_size"] = int(np.min(counts))
            metrics["max_cluster_size"] = int(np.max(counts))
            metrics["cluster_size_std"] = float(np.std(counts))
            metrics["cluster_balance"] = float(np.min(counts) / np.max(counts))
            
            # Intra-cluster distances (compactness)
            intra_distances = []
            for label in unique_labels:
                cluster_points = X[labels == label]
                if len(cluster_points) > 1:
                    # Average pairwise distance within cluster
                    distances = pairwise_distances(cluster_points)
                    # Get upper triangle (excluding diagonal)
                    upper_tri = np.triu(distances, k=1)
                    non_zero_distances = upper_tri[upper_tri > 0]
                    if len(non_zero_distances) > 0:
                        intra_distances.append(np.mean(non_zero_distances))
            
            if intra_distances:
                metrics["avg_intra_cluster_distance"] = float(np.mean(intra_distances))
                metrics["std_intra_cluster_distance"] = float(np.std(intra_distances))
            
            # Inter-cluster distances (separation)
            if centers is not None and len(centers) > 1:
                inter_distances = pairwise_distances(centers)
                # Get upper triangle (excluding diagonal)
                upper_tri = np.triu(inter_distances, k=1)
                non_zero_distances = upper_tri[upper_tri > 0]
                if len(non_zero_distances) > 0:
                    metrics["avg_inter_cluster_distance"] = float(np.mean(non_zero_distances))
                    metrics["min_inter_cluster_distance"] = float(np.min(non_zero_distances))
            
        except Exception as e:
            logger.warning(f"Error calculating quality metrics: {e}")
        
        return metrics
    
    def _calculate_kmeans_metrics(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        centers: np.ndarray
    ) -> Dict[str, float]:
        """Calculate K-Means specific metrics."""
        metrics = {}
        
        try:
            # Within-cluster sum of squares (WCSS) - same as inertia
            wcss = 0
            for i, center in enumerate(centers):
                cluster_points = X[labels == i]
                if len(cluster_points) > 0:
                    wcss += np.sum((cluster_points - center) ** 2)
            
            metrics["wcss"] = float(wcss)
            metrics["inertia"] = float(wcss)  # Alias for WCSS
            
            # Between-cluster sum of squares (BCSS)
            overall_center = np.mean(X, axis=0)
            bcss = 0
            for i, center in enumerate(centers):
                cluster_size = np.sum(labels == i)
                bcss += cluster_size * np.sum((center - overall_center) ** 2)
            
            metrics["bcss"] = float(bcss)
            
            # Total sum of squares (TSS)
            tss = np.sum((X - overall_center) ** 2)
            metrics["tss"] = float(tss)
            
            # Explained variance ratio
            if tss > 0:
                metrics["explained_variance_ratio"] = float(bcss / tss)
            
        except Exception as e:
            logger.warning(f"Error calculating K-Means metrics: {e}")
        
        return metrics
    
    def calculate_silhouette_analysis(
        self,
        X: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, Any]:
        """
        Detailed silhouette analysis for each cluster.
        
        Args:
            X: Feature matrix
            labels: Cluster labels
            
        Returns:
            Dictionary with detailed silhouette analysis
        """
        try:
            # Overall silhouette score
            overall_silhouette = silhouette_score(X, labels)
            
            # Per-sample silhouette scores
            sample_silhouette_values = silhouette_samples(X, labels)
            
            # Per-cluster analysis
            cluster_analysis = {}
            for cluster_id in np.unique(labels):
                cluster_silhouettes = sample_silhouette_values[labels == cluster_id]
                
                cluster_analysis[int(cluster_id)] = {
                    "mean_silhouette": float(np.mean(cluster_silhouettes)),
                    "std_silhouette": float(np.std(cluster_silhouettes)),
                    "min_silhouette": float(np.min(cluster_silhouettes)),
                    "max_silhouette": float(np.max(cluster_silhouettes)),
                    "size": int(len(cluster_silhouettes)),
                    "below_average_count": int(np.sum(cluster_silhouettes < overall_silhouette))
                }
            
            return {
                "overall_silhouette": float(overall_silhouette),
                "sample_silhouette_values": sample_silhouette_values,
                "cluster_analysis": cluster_analysis,
                "clusters_above_average": int(sum(
                    1 for analysis in cluster_analysis.values()
                    if analysis["mean_silhouette"] > overall_silhouette
                ))
            }
            
        except Exception as e:
            logger.error(f"Error in silhouette analysis: {e}")
            return {"error": str(e)}
    
    def evaluate_cluster_stability(
        self,
        X: np.ndarray,
        algorithm_func,
        k: int,
        n_iterations: int = 10,
        sample_ratio: float = 0.8
    ) -> Dict[str, float]:
        """
        Evaluate clustering stability using bootstrap sampling.
        
        Args:
            X: Feature matrix
            algorithm_func: Function that returns cluster labels
            k: Number of clusters
            n_iterations: Number of bootstrap iterations
            sample_ratio: Ratio of samples to use in each iteration
            
        Returns:
            Dictionary with stability metrics
        """
        logger.info(f"Evaluating cluster stability with {n_iterations} iterations")
        
        n_samples = X.shape[0]
        sample_size = int(n_samples * sample_ratio)
        
        # Store labels from each iteration
        all_labels = []
        
        for i in range(n_iterations):
            # Bootstrap sample
            indices = np.random.choice(n_samples, sample_size, replace=False)
            X_sample = X[indices]
            
            try:
                # Get cluster labels
                labels = algorithm_func(X_sample, k)
                all_labels.append((indices, labels))
            except Exception as e:
                logger.warning(f"Iteration {i} failed: {e}")
                continue
        
        if len(all_labels) < 2:
            return {"error": "insufficient_iterations"}
        
        # Calculate stability metrics
        stability_scores = []
        
        for i in range(len(all_labels)):
            for j in range(i + 1, len(all_labels)):
                indices_i, labels_i = all_labels[i]
                indices_j, labels_j = all_labels[j]
                
                # Find common samples
                common_indices = np.intersect1d(indices_i, indices_j)
                
                if len(common_indices) > 1:
                    # Get labels for common samples
                    mask_i = np.isin(indices_i, common_indices)
                    mask_j = np.isin(indices_j, common_indices)
                    
                    common_labels_i = labels_i[mask_i]
                    common_labels_j = labels_j[mask_j]
                    
                    # Calculate adjusted rand index
                    from sklearn.metrics import adjusted_rand_score
                    ari = adjusted_rand_score(common_labels_i, common_labels_j)
                    stability_scores.append(ari)
        
        if stability_scores:
            return {
                "mean_stability": float(np.mean(stability_scores)),
                "std_stability": float(np.std(stability_scores)),
                "min_stability": float(np.min(stability_scores)),
                "max_stability": float(np.max(stability_scores)),
                "n_comparisons": len(stability_scores)
            }
        else:
            return {"error": "no_valid_comparisons"}
    
    def compare_clusterings(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compare multiple clustering results.
        
        Args:
            results: List of clustering evaluation results
            
        Returns:
            Comparison analysis
        """
        if len(results) < 2:
            return {"error": "need_at_least_two_results"}
        
        comparison = {
            "n_results": len(results),
            "metrics_comparison": {},
            "rankings": {}
        }
        
        # Extract metrics for comparison
        metric_names = ["silhouette", "davies_bouldin", "calinski_harabasz"]
        
        for metric in metric_names:
            values = [result.get(metric) for result in results if metric in result]
            
            if values:
                comparison["metrics_comparison"][metric] = {
                    "values": values,
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values))
                }
                
                # Ranking (higher is better for silhouette and calinski_harabasz)
                if metric in ["silhouette", "calinski_harabasz"]:
                    ranking = np.argsort(values)[::-1]  # Descending order
                else:  # davies_bouldin (lower is better)
                    ranking = np.argsort(values)  # Ascending order
                
                comparison["rankings"][metric] = ranking.tolist()
        
        return comparison
