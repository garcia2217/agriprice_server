"""
Clustering Algorithm Manager

This module provides a unified interface for different clustering algorithms
including K-Means, Fuzzy C-Means, and Spectral Clustering.
"""

import logging
import time
from typing import Dict, Any, Tuple, Optional
import numpy as np
from dataclasses import dataclass

# Scikit-learn imports
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler

# Fuzzy C-Means (optional)
try:
    import skfuzzy as fuzz
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
    logging.warning("scikit-fuzzy not available. Fuzzy C-Means will be skipped.")

logger = logging.getLogger(__name__)


@dataclass
class ClusteringResult:
    """
    Container for clustering algorithm results.
    
    Attributes:
        algorithm: Name of the clustering algorithm
        k: Number of clusters
        labels: Cluster assignment labels
        model: Fitted clustering model (if applicable)
        execution_time: Time taken to fit the model in seconds
        centers: Cluster centers (if available)
        membership_matrix: Fuzzy membership matrix (for FCM only)
        additional_info: Any additional algorithm-specific information
    """
    algorithm: str
    k: int
    labels: np.ndarray
    model: Any
    execution_time: float
    centers: Optional[np.ndarray] = None
    membership_matrix: Optional[np.ndarray] = None
    additional_info: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.additional_info is None:
            self.additional_info = {}


class ClusteringAlgorithmManager:
    """
    Manages different clustering algorithms with unified interface.
    
    Supports:
    - K-Means clustering
    - Fuzzy C-Means clustering (if scikit-fuzzy available)
    - Spectral clustering
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize algorithm manager.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.supported_algorithms = ['kmeans', 'spectral']
        
        if FUZZY_AVAILABLE:
            self.supported_algorithms.append('fcm')
        
        logger.info(f"Initialized algorithm manager with algorithms: {self.supported_algorithms}")
    
    def fit_clustering(
        self,
        X: np.ndarray,
        algorithm: str,
        k: int,
        algorithm_params: Optional[Dict[str, Any]] = None
    ) -> ClusteringResult:
        """
        Fit clustering algorithm to data.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            algorithm: Algorithm name ('kmeans', 'fcm', 'spectral')
            k: Number of clusters
            algorithm_params: Algorithm-specific parameters
            
        Returns:
            ClusteringResult object with fitted model and results
            
        Raises:
            ValueError: If algorithm is not supported or parameters are invalid
        """
        if algorithm not in self.supported_algorithms:
            raise ValueError(f"Algorithm '{algorithm}' not supported. Available: {self.supported_algorithms}")
        
        if k < 2:
            raise ValueError(f"Number of clusters must be at least 2, got {k}")
        
        if k >= X.shape[0]:
            raise ValueError(f"Number of clusters ({k}) must be less than number of samples ({X.shape[0]})")
        
        algorithm_params = algorithm_params or {}
        
        logger.info(f"Fitting {algorithm} clustering with k={k} on data shape {X.shape}")
        
        # Route to appropriate algorithm
        if algorithm == 'kmeans':
            return self._fit_kmeans(X, k, algorithm_params)
        elif algorithm == 'fcm':
            return self._fit_fcm(X, k, algorithm_params)
        elif algorithm == 'spectral':
            return self._fit_spectral(X, k, algorithm_params)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def _fit_kmeans(
        self,
        X: np.ndarray,
        k: int,
        params: Dict[str, Any]
    ) -> ClusteringResult:
        """
        Fit K-Means clustering.
        
        Args:
            X: Feature matrix
            k: Number of clusters
            params: K-Means parameters
            
        Returns:
            ClusteringResult for K-Means
        """
        # Default parameters
        default_params = {
            'n_init': 10,
            'max_iter': 300,
            'tol': 1e-4,
            'algorithm': 'lloyd'
        }
        default_params.update(params)
        
        # Create and fit model
        model = KMeans(
            n_clusters=k,
            random_state=self.random_state,
            **default_params
        )
        
        start_time = time.time()
        labels = model.fit_predict(X)
        execution_time = time.time() - start_time
        
        # Extract results
        centers = model.cluster_centers_
        inertia = model.inertia_
        
        additional_info = {
            'inertia': inertia,
            'n_iter': model.n_iter_,
            'converged': model.n_iter_ < default_params['max_iter']
        }
        
        logger.info(f"K-Means completed in {execution_time:.3f}s, inertia: {inertia:.3f}")
        
        return ClusteringResult(
            algorithm='kmeans',
            k=k,
            labels=labels,
            model=model,
            execution_time=execution_time,
            centers=centers,
            additional_info=additional_info
        )
    
    def _fit_fcm(
        self,
        X: np.ndarray,
        k: int,
        params: Dict[str, Any]
    ) -> ClusteringResult:
        """
        Fit Fuzzy C-Means clustering.
        
        Args:
            X: Feature matrix
            k: Number of clusters
            params: FCM parameters
            
        Returns:
            ClusteringResult for FCM
        """
        if not FUZZY_AVAILABLE:
            raise ValueError("Fuzzy C-Means not available. Install scikit-fuzzy.")
        
        # Default parameters
        default_params = {
            'm': 2.0,  # Fuzziness parameter
            'error': 1e-5,
            'maxiter': 150
        }
        default_params.update(params)
        
        # Transpose data for scikit-fuzzy (expects features × samples)
        X_T = X.T
        
        start_time = time.time()
        
        # Fit FCM
        centers, membership_matrix, _, _, _, n_iter, fpc = fuzz.cluster.cmeans(
            X_T,
            k,
            default_params['m'],
            error=default_params['error'],
            maxiter=default_params['maxiter'],
            seed=self.random_state
        )
        
        execution_time = time.time() - start_time
        
        # Get hard cluster assignments (highest membership)
        labels = np.argmax(membership_matrix, axis=0)
        
        additional_info = {
            'fpc': fpc,  # Fuzzy partition coefficient
            'n_iter': n_iter,
            'converged': n_iter < default_params['maxiter'],
            'fuzziness_parameter': default_params['m']
        }
        
        logger.info(f"FCM completed in {execution_time:.3f}s, FPC: {fpc:.3f}")
        
        return ClusteringResult(
            algorithm='fcm',
            k=k,
            labels=labels,
            model=None,  # FCM doesn't have a persistent model object
            execution_time=execution_time,
            centers=centers.T,  # Transpose back to samples × features
            membership_matrix=membership_matrix.T,  # Transpose to samples × clusters
            additional_info=additional_info
        )
    
    def _fit_spectral(
        self,
        X: np.ndarray,
        k: int,
        params: Dict[str, Any]
    ) -> ClusteringResult:
        """
        Fit Spectral clustering.
        
        Args:
            X: Feature matrix
            k: Number of clusters
            params: Spectral clustering parameters
            
        Returns:
            ClusteringResult for Spectral clustering
        """
        # Default parameters
        default_params = {
            'affinity': 'rbf',
            'gamma': 1.0,
            'n_init': 10,
            'assign_labels': 'kmeans'
        }
        default_params.update(params)
        
        # Create and fit model
        model = SpectralClustering(
            n_clusters=k,
            random_state=self.random_state,
            **default_params
        )
        
        start_time = time.time()
        labels = model.fit_predict(X)
        execution_time = time.time() - start_time
        
        additional_info = {
            'affinity_matrix_shape': getattr(model, 'affinity_matrix_', np.array([])).shape,
            'n_features_in': getattr(model, 'n_features_in_', X.shape[1])
        }
        
        logger.info(f"Spectral clustering completed in {execution_time:.3f}s")
        
        return ClusteringResult(
            algorithm='spectral',
            k=k,
            labels=labels,
            model=model,
            execution_time=execution_time,
            centers=None,  # Spectral clustering doesn't provide explicit centers
            additional_info=additional_info
        )
    
    def predict(
        self,
        result: ClusteringResult,
        X_new: np.ndarray
    ) -> np.ndarray:
        """
        Predict cluster labels for new data points.
        
        Args:
            result: Previous clustering result
            X_new: New data points to predict
            
        Returns:
            Predicted cluster labels
            
        Note:
            Only works for algorithms that support prediction (K-Means).
            For other algorithms, raises NotImplementedError.
        """
        if result.algorithm == 'kmeans' and result.model is not None:
            return result.model.predict(X_new)
        elif result.algorithm == 'fcm' and result.centers is not None:
            # For FCM, use distance to centers for prediction
            from scipy.spatial.distance import cdist
            distances = cdist(X_new, result.centers)
            return np.argmin(distances, axis=1)
        else:
            raise NotImplementedError(
                f"Prediction not supported for {result.algorithm} clustering"
            )
    
    def get_supported_algorithms(self) -> list:
        """
        Get list of supported clustering algorithms.
        
        Returns:
            List of supported algorithm names
        """
        return self.supported_algorithms.copy()
    
    def validate_algorithm_params(
        self,
        algorithm: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate and clean algorithm parameters.
        
        Args:
            algorithm: Algorithm name
            params: Parameters to validate
            
        Returns:
            Dictionary with validation results:
            {
                'valid': bool,
                'errors': List[str],
                'warnings': List[str],
                'cleaned_params': Dict[str, Any]
            }
        """
        errors = []
        warnings = []
        cleaned_params = params.copy()
        
        if algorithm == 'kmeans':
            # Validate K-Means parameters
            if 'n_init' in params and params['n_init'] < 1:
                errors.append("n_init must be at least 1")
            
            if 'max_iter' in params and params['max_iter'] < 1:
                errors.append("max_iter must be at least 1")
            
            if 'tol' in params and params['tol'] <= 0:
                errors.append("tol must be positive")
        
        elif algorithm == 'fcm':
            # Validate FCM parameters
            if 'm' in params and params['m'] <= 1:
                errors.append("Fuzziness parameter 'm' must be greater than 1")
            
            if 'error' in params and params['error'] <= 0:
                errors.append("error tolerance must be positive")
            
            if 'maxiter' in params and params['maxiter'] < 1:
                errors.append("maxiter must be at least 1")
        
        elif algorithm == 'spectral':
            # Validate Spectral clustering parameters
            valid_affinities = ['nearest_neighbors', 'rbf', 'polynomial', 'sigmoid', 'cosine']
            if 'affinity' in params and params['affinity'] not in valid_affinities:
                if not callable(params['affinity']):
                    errors.append(f"affinity must be one of {valid_affinities} or callable")
            
            if 'gamma' in params and params['gamma'] <= 0:
                errors.append("gamma must be positive")
            
            if 'n_init' in params and params['n_init'] < 1:
                errors.append("n_init must be at least 1")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'cleaned_params': cleaned_params
        }
