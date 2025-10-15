"""
API Response Formatter for Clustering Results

This module formats clustering results for API responses and various output formats.
"""

import logging
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class APIResponseFormatter:
    """
    Formats clustering results for API responses and different output formats.
    
    Supports multiple output formats:
    - API format: List of city assignments with coordinates
    - Detailed format: Comprehensive results with metrics
    - Simple format: Basic clustering results
    """
    
    def __init__(self, coordinates_data: pd.DataFrame):
        """
        Initialize API response formatter.
        
        Args:
            coordinates_data: DataFrame with City, Latitude, Longitude columns
        """
        self.coordinates_data = coordinates_data
        logger.info(f"Initialized API formatter with coordinates for {len(coordinates_data)} cities")
    
    def format_monthly_trends(
        self,
        labels: np.ndarray,
        cities: List[str],
        preprocessed_df: pd.DataFrame,
        commodities: Optional[List[str]] = None,
        years: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Build monthly average price trends per cluster and commodity.
        Assumes preprocessed_df has columns: City, Commodity, Date, Price.
        Returns dict: { commodity: [ {clusterId, data: [m1, m2, ...]}, ... ] }
        where data is month-by-month averages across the requested years (12 * n_years values).
        """
        if preprocessed_df is None or preprocessed_df.empty:
            return {}

        df = preprocessed_df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month

        # Select commodities/years if provided
        if commodities:
            df = df[df['Commodity'].isin(commodities)]
        else:
            commodities = sorted(df['Commodity'].dropna().unique().tolist())

        if years:
            df = df[df['Year'].isin(years)]
        else:
            years = sorted(df['Year'].dropna().unique().tolist())

        # Map each city to cluster
        city_to_cluster = {city: int(label) for city, label in zip(cities, labels)}
        df['clusterId'] = df['City'].map(city_to_cluster)

        # Month order across years (e.g., 2020-01 .. 2020-12, 2021-01 .. 2021-12)
        months_order = [(y, m) for y in years for m in range(1, 13)]

        trends: Dict[str, List[Dict[str, Any]]] = {}
        for commodity in commodities:
            com_df = df[df['Commodity'] == commodity]
            if com_df.empty:
                continue

            cluster_ids = sorted([int(c) for c in com_df['clusterId'].dropna().unique().tolist()])
            series_per_cluster: List[Dict[str, Any]] = []

            for cid in cluster_ids:
                cdf = com_df[com_df['clusterId'] == cid]

                # Monthly avg across all cities in cluster for each (year, month)
                month_values: List[Optional[float]] = []
                for (y, m) in months_order:
                    val = cdf[(cdf['Year'] == y) & (cdf['Month'] == m)]['Price'].mean()
                    month_values.append(float(val) if not np.isnan(val) else None)

                series_per_cluster.append({
                    "clusterId": cid,
                    "data": month_values
                })

            if series_per_cluster:
                trends[commodity] = series_per_cluster

        return trends

    def format_frontend_response(
        self,
        analysis_id: str,
        labels: np.ndarray,
        cities: List[str],
        preprocessed_df: Optional[pd.DataFrame] = None,
        commodities: Optional[List[str]] = None,
        years: Optional[List[int]] = None,
        clusters_palette: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Assemble response.json-like structure for the frontend map visualization.
        Shape:
        {
          analysis_id, years, clusters: [{id,name,color,bgColor,hexColor}],
          cities: [{name,lat,lon,clusterId}],
          trends: { commodity: [ {clusterId, data: [...]}, ... ] }
        }
        """
        # Cities with coordinates and cluster IDs
        assignments = self.format_cluster_assignments(cities, labels)

        # Years list (string) from preprocessed_df if present or provided
        years_list: List[str] = []
        if preprocessed_df is not None and not preprocessed_df.empty and 'Date' in preprocessed_df.columns:
            y = sorted(pd.to_datetime(preprocessed_df['Date']).dt.year.dropna().unique().tolist())
            if years:
                y = [yy for yy in y if yy in years]
            years_list = [str(yy) for yy in y]
        elif years:
            years_list = [str(yy) for yy in sorted(years)]

        # Prepare cluster palette (10 colors by default for cluster IDs 0..9)
        unique_clusters = sorted(list(set(int(l) for l in labels)))
        if not clusters_palette:
            default_palette = [
                {"color": "text-red-500", "bgColor": "bg-red-500", "hexColor": "#EF4444"},
                {"color": "text-green-500", "bgColor": "bg-green-500", "hexColor": "#22C55E"},
                {"color": "text-yellow-500", "bgColor": "bg-yellow-500", "hexColor": "#EAB308"},
                {"color": "text-blue-500", "bgColor": "bg-blue-500", "hexColor": "#3B82F6"},
                {"color": "text-purple-500", "bgColor": "bg-purple-500", "hexColor": "#8B5CF6"},
                {"color": "text-orange-500", "bgColor": "bg-orange-500", "hexColor": "#F97316"},
                {"color": "text-teal-500", "bgColor": "bg-teal-500", "hexColor": "#14B8A6"},
                {"color": "text-pink-500", "bgColor": "bg-pink-500", "hexColor": "#EC4899"},
                {"color": "text-indigo-500", "bgColor": "bg-indigo-500", "hexColor": "#6366F1"},
                {"color": "text-gray-500", "bgColor": "bg-gray-500", "hexColor": "#6B7280"}
            ]
            clusters_palette = []
            for i, cid in enumerate(unique_clusters):
                c = default_palette[i % len(default_palette)]
                clusters_palette.append({"id": cid, "name": f"Klaster {cid}", **c})

        # Trends (optional if preprocessed_df provided)
        trends: Dict[str, Any] = {}
        if preprocessed_df is not None and not preprocessed_df.empty:
            trends = self.format_monthly_trends(
                labels=labels,
                cities=cities,
                preprocessed_df=preprocessed_df,
                commodities=commodities,
                years=[int(y) for y in years_list] if years_list else None
            )

        return {
            "analysis_id": analysis_id,
            "years": years_list,
            "clusters": clusters_palette,
            "cities": assignments,
            "trends": trends
        }
    def format_cluster_assignments(
        self,
        cities: List[str],
        labels: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Format cluster assignments for API response.
        
        Args:
            cities: List of city names
            labels: Cluster labels array
            
        Returns:
            List of dictionaries with name, lat, lon, clusterId format:
            [
                {"name": "Jakarta", "lat": -6.175, "lon": 106.828, "clusterId": 0},
                {"name": "Bandung", "lat": -6.917, "lon": 107.619, "clusterId": 1},
                ...
            ]
        """
        if len(cities) != len(labels):
            raise ValueError(f"Number of cities ({len(cities)}) doesn't match number of labels ({len(labels)})")
        
        result = []
        
        for city, cluster_id in zip(cities, labels):
            # Get coordinates for city
            city_coords = self.coordinates_data[
                self.coordinates_data['City'] == city
            ]
            
            if not city_coords.empty:
                lat = float(city_coords.iloc[0]['Latitude'])
                lon = float(city_coords.iloc[0]['Longitude'])
            else:
                # Log missing coordinates but don't fail
                logger.warning(f"No coordinates found for city: {city}")
                lat, lon = None, None
            
            result.append({
                "name": city,
                "lat": lat,
                "lon": lon,
                "clusterId": int(cluster_id)
            })
        
        logger.info(f"Formatted {len(result)} city assignments for API response")
        return result
    
    def format_detailed_response(
        self,
        cities: List[str],
        labels: np.ndarray,
        metrics: Dict[str, float],
        algorithm: str,
        k: int,
        execution_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Format comprehensive API response with metadata.
        
        Args:
            cities: List of city names
            labels: Cluster labels array
            metrics: Clustering evaluation metrics
            algorithm: Algorithm name
            k: Number of clusters
            execution_time: Algorithm execution time in seconds
            
        Returns:
            Comprehensive response dictionary with assignments, metrics, and metadata
        """
        assignments = self.format_cluster_assignments(cities, labels)
        
        # Calculate cluster statistics
        cluster_sizes = pd.Series(labels).value_counts().sort_index()
        
        response = {
            "assignments": assignments,
            "metrics": {
                "silhouette_score": round(metrics.get("silhouette", 0), 4),
                "davies_bouldin_index": round(metrics.get("davies_bouldin", 0), 4),
                "calinski_harabasz_index": round(metrics.get("calinski_harabasz", 0), 4)
            },
            "metadata": {
                "algorithm": algorithm,
                "k": k,
                "total_cities": len(cities),
                "cluster_sizes": {
                    str(cluster_id): int(count)
                    for cluster_id, count in cluster_sizes.items()
                }
            }
        }
        
        # Add execution time if provided
        if execution_time is not None:
            response["metadata"]["execution_time_seconds"] = round(execution_time, 4)
        
        # Add additional metrics if available
        if "inertia" in metrics:
            response["metrics"]["inertia"] = round(metrics["inertia"], 4)
        
        logger.info(f"Formatted detailed response for {algorithm} with k={k}")
        return response
    
    def format_simple_response(
        self,
        cities: List[str],
        labels: np.ndarray,
        algorithm: str,
        k: int
    ) -> Dict[str, Any]:
        """
        Format simple clustering response.
        
        Args:
            cities: List of city names
            labels: Cluster labels array
            algorithm: Algorithm name
            k: Number of clusters
            
        Returns:
            Simple response dictionary with basic clustering information
        """
        cluster_assignments = {
            city: int(label) for city, label in zip(cities, labels)
        }
        
        cluster_sizes = pd.Series(labels).value_counts().sort_index()
        
        return {
            "algorithm": algorithm,
            "k": k,
            "cluster_assignments": cluster_assignments,
            "cluster_sizes": cluster_sizes.to_dict(),
            "total_cities": len(cities)
        }
    
    def format_cluster_summary(
        self,
        cities: List[str],
        labels: np.ndarray
    ) -> Dict[str, Any]:
        """
        Create summary statistics for each cluster.
        
        Args:
            cities: List of city names
            labels: Cluster labels array
            
        Returns:
            Dictionary with cluster summary information
        """
        cluster_summary = {}
        
        for cluster_id in np.unique(labels):
            cluster_cities = [city for city, label in zip(cities, labels) if label == cluster_id]
            
            # Get coordinates for cluster cities
            cluster_coords = self.coordinates_data[
                self.coordinates_data['City'].isin(cluster_cities)
            ]
            
            summary = {
                "cities": cluster_cities,
                "city_count": len(cluster_cities),
                "cities_with_coordinates": len(cluster_coords)
            }
            
            # Calculate geographic center if coordinates available
            if not cluster_coords.empty:
                summary.update({
                    "geographic_center": {
                        "lat": float(cluster_coords['Latitude'].mean()),
                        "lon": float(cluster_coords['Longitude'].mean())
                    },
                    "geographic_bounds": {
                        "north": float(cluster_coords['Latitude'].max()),
                        "south": float(cluster_coords['Latitude'].min()),
                        "east": float(cluster_coords['Longitude'].max()),
                        "west": float(cluster_coords['Longitude'].min())
                    }
                })
            
            cluster_summary[str(cluster_id)] = summary
        
        return cluster_summary
    
    def format_comparison_response(
        self,
        results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Format comparison response across multiple algorithms or k values.
        
        Args:
            results: Dictionary of results keyed by algorithm or configuration
            
        Returns:
            Comparison response with best configurations and metrics
        """
        comparison = {
            "results": results,
            "best_configurations": self._identify_best_configurations(results),
            "summary": {
                "total_configurations": len(results),
                "algorithms_tested": list(set(
                    result.get("metadata", {}).get("algorithm", "unknown")
                    for result in results.values()
                )),
                "k_values_tested": list(set(
                    result.get("metadata", {}).get("k", 0)
                    for result in results.values()
                ))
            }
        }
        
        return comparison
    
    def _identify_best_configurations(
        self,
        results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Identify best configurations based on clustering metrics.
        
        Args:
            results: Dictionary of clustering results
            
        Returns:
            Dictionary with best configurations for different metrics
        """
        if not results:
            return {}
        
        best_configs = {}
        
        # Find best silhouette score (higher is better)
        silhouette_scores = {
            config: result.get("metrics", {}).get("silhouette_score", -1)
            for config, result in results.items()
        }
        if silhouette_scores:
            best_silhouette = max(silhouette_scores.items(), key=lambda x: x[1])
            best_configs["best_silhouette"] = {
                "configuration": best_silhouette[0],
                "score": best_silhouette[1]
            }
        
        # Find best Davies-Bouldin index (lower is better)
        db_scores = {
            config: result.get("metrics", {}).get("davies_bouldin_index", float('inf'))
            for config, result in results.items()
        }
        if db_scores:
            best_db = min(db_scores.items(), key=lambda x: x[1])
            best_configs["best_davies_bouldin"] = {
                "configuration": best_db[0],
                "score": best_db[1]
            }
        
        # Find best Calinski-Harabasz index (higher is better)
        ch_scores = {
            config: result.get("metrics", {}).get("calinski_harabasz_index", 0)
            for config, result in results.items()
        }
        if ch_scores:
            best_ch = max(ch_scores.items(), key=lambda x: x[1])
            best_configs["best_calinski_harabasz"] = {
                "configuration": best_ch[0],
                "score": best_ch[1]
            }
        
        return best_configs
    
    def format_error_response(
        self,
        error_message: str,
        error_type: str = "ClusteringError",
        details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Format error response for API.
        
        Args:
            error_message: Error message
            error_type: Type of error
            details: Additional error details
            
        Returns:
            Formatted error response
        """
        error_response = {
            "success": False,
            "error": {
                "type": error_type,
                "message": error_message
            }
        }
        
        if details:
            error_response["error"]["details"] = details
        
        return error_response
    
    def format_success_response(
        self,
        data: Any,
        message: str = "Clustering completed successfully"
    ) -> Dict[str, Any]:
        """
        Format success response for API.
        
        Args:
            data: Response data
            message: Success message
            
        Returns:
            Formatted success response
        """
        return {
            "success": True,
            "message": message,
            "data": data
        }
