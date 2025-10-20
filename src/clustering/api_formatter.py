"""
API Response Formatter for Clustering Results

This module formats clustering results for API responses and various output formats.
"""

import logging
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score, davies_bouldin_score

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

    def format_radar_features(
        self,
        merged_df: pd.DataFrame,
        labels: np.ndarray,
        cities: List[str]
    ) -> Dict[str, Any]:
        """
        Calculate radar chart features from preprocessed data with cluster labels.
        
        Args:
            merged_df: Preprocessed DataFrame with cluster labels already added
            labels: Cluster assignment labels
            cities: List of city names
            
        Returns:
            Dictionary with radar features: {cluster_id: {commodity: normalized_value}}
        """
        if merged_df is None or merged_df.empty:
            return {}
        
        # Check if we have the required columns
        required_cols = ['Cluster', 'Commodity', 'Price']
        if not all(col in merged_df.columns for col in required_cols):
            logger.warning("Missing required columns for radar features calculation")
            return {}
        
        # Get unique commodities
        commodities = merged_df['Commodity'].unique()
        if len(commodities) < 3:
            logger.info(f"Less than 3 commodities ({len(commodities)}), skipping radar features")
            return {}
        
        # Calculate average price per cluster and commodity
        cluster_commodity_avg = merged_df.groupby(['Cluster', 'Commodity'])['Price'].mean()
        
        # Normalize per commodity across clusters (0-1 scale)
        radar_features = {}
        
        for cluster_id in sorted(merged_df['Cluster'].unique()):
            cluster_data = {}
            
            for commodity in commodities:
                # Get all values for this commodity across all clusters
                commodity_values = []
                for cid in merged_df['Cluster'].unique():
                    if (cid, commodity) in cluster_commodity_avg.index:
                        commodity_values.append(cluster_commodity_avg[(cid, commodity)])
                
                if len(commodity_values) > 1:  # Need at least 2 clusters to normalize
                    min_val = min(commodity_values)
                    max_val = max(commodity_values)
                    
                    if max_val > min_val:  # Avoid division by zero
                        current_val = cluster_commodity_avg.get((cluster_id, commodity), 0)
                        normalized_val = (current_val - min_val) / (max_val - min_val)
                    else:
                        normalized_val = 0.5  # All values are the same
                else:
                    normalized_val = 0.5  # Only one cluster
                
                cluster_data[commodity] = round(normalized_val, 3)
            
            radar_features[str(cluster_id)] = cluster_data
        
        logger.info(f"Calculated radar features for {len(radar_features)} clusters and {len(commodities)} commodities")
        return radar_features

    def format_boxplot_data(
        self,
        merged_df: pd.DataFrame,
        labels: np.ndarray,
        cities: List[str]
    ) -> Dict[str, Any]:
        """
        Calculate boxplot data from preprocessed data with cluster labels.
        
        Args:
            merged_df: Preprocessed DataFrame with cluster labels already added
            labels: Cluster assignment labels
            cities: List of city names
            
        Returns:
            Dictionary with boxplot data structure
        """
        if merged_df is None or merged_df.empty:
            return {}
        
        # Check if we have the required columns
        required_cols = ['Cluster', 'Commodity', 'Price', 'Date']
        if not all(col in merged_df.columns for col in required_cols):
            logger.warning("Missing required columns for boxplot data calculation")
            return {}
        
        # Ensure Date is datetime
        df = merged_df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df['Year'] = df['Date'].dt.year
        
        # Get unique values and convert to native Python types
        commodities = [str(c) for c in sorted(df['Commodity'].unique())]
        clusters = [int(c) for c in sorted(df['Cluster'].unique())]
        years = [int(y) for y in sorted(df['Year'].unique())]
        
        if len(commodities) == 0 or len(clusters) == 0 or len(years) == 0:
            logger.warning("Insufficient data for boxplot calculation")
            return {}
        
        # Initialize data structure
        data = {}
        statistics = {}
        
        for commodity in commodities:
            data[commodity] = {}
            statistics[commodity] = {}
            
            for year in years:
                data[commodity][str(year)] = {}
                statistics[commodity][str(year)] = {}
                
                for cluster in clusters:
                    # Get prices for this commodity-year-cluster combination
                    cluster_data = df[
                        (df['Commodity'] == commodity) & 
                        (df['Year'] == year) & 
                        (df['Cluster'] == cluster)
                    ]['Price'].tolist()
                    
                    # Store raw prices
                    data[commodity][str(year)][str(cluster)] = cluster_data
                    
                    # Calculate statistics if we have data
                    if cluster_data:
                        prices = np.array(cluster_data)
                        stats = self._calculate_boxplot_statistics(prices)
                        statistics[commodity][str(year)][str(cluster)] = stats
                    else:
                        # Empty statistics for missing data
                        statistics[commodity][str(year)][str(cluster)] = {
                            "min": None, "q1": None, "median": None, "q3": None,
                            "max": None, "mean": None, "std": None, "outliers": []
                        }
        
        # Get cluster colors from existing palette
        cluster_colors = {}
        for i, cluster in enumerate(clusters):
            # Use the same color palette as clusters
            default_palette = [
                "#EF4444", "#22C55E", "#EAB308", "#3B82F6", "#8B5CF6",
                "#F97316", "#14B8A6", "#EC4899", "#6366F1", "#6B7280"
            ]
            cluster_colors[str(cluster)] = default_palette[i % len(default_palette)]
        
        boxplot_data = {
            "commodities": commodities,
            "clusters": [int(c) for c in clusters],
            "years": years,
            "data": data,
            "statistics": statistics,
            "clusterColors": cluster_colors
        }
        
        logger.info(f"Calculated boxplot data for {len(commodities)} commodities, {len(clusters)} clusters, {len(years)} years")
        return boxplot_data
    
    def _calculate_boxplot_statistics(self, prices: np.ndarray) -> Dict[str, Any]:
        """Calculate boxplot statistics for a price array."""
        if len(prices) == 0:
            return {
                "min": None, "q1": None, "median": None, "q3": None,
                "max": None, "mean": None, "std": None, "outliers": []
            }
        
        # Basic statistics
        min_val = float(np.min(prices))
        max_val = float(np.max(prices))
        mean_val = float(np.mean(prices))
        std_val = float(np.std(prices))
        
        # Quartiles
        q1 = float(np.percentile(prices, 25))
        median = float(np.median(prices))
        q3 = float(np.percentile(prices, 75))
        
        # Outliers using IQR method
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = [float(x) for x in prices if x < lower_bound or x > upper_bound]
        
        return {
            "min": min_val,
            "q1": q1,
            "median": median,
            "q3": q3,
            "max": max_val,
            "mean": mean_val,
            "std": std_val,
            "outliers": outliers
        }

    def format_correlation_matrix(
        self,
        merged_df: pd.DataFrame,
        config: Any
    ) -> Dict[str, Any]:
        """
        Calculate correlation matrix for commodity prices.
        
        Args:
            merged_df: Preprocessed DataFrame with cluster labels
            config: Configuration object with correlation_mode setting
            
        Returns:
            Dictionary with correlation matrix data
        """
        if merged_df is None or merged_df.empty:
            return {}
        
        # Check if we have the required columns
        required_cols = ['Commodity', 'Price', 'Date']
        if not all(col in merged_df.columns for col in required_cols):
            logger.warning("Missing required columns for correlation calculation")
            return {}
        
        # Get unique commodities
        commodities = sorted(merged_df['Commodity'].unique())
        if len(commodities) < 2:
            logger.warning("Need at least 2 commodities for correlation calculation")
            return {}
        
        # Route to appropriate calculation method
        correlation_mode = getattr(config, 'correlation_mode', 'global')
        
        if correlation_mode == "global":
            return self._calculate_global_correlation(merged_df, commodities)
        elif correlation_mode == "per_cluster":
            return self._calculate_per_cluster_correlation(merged_df, commodities)
        else:
            logger.warning(f"Unknown correlation mode: {correlation_mode}")
            return {}
    
    def _calculate_global_correlation(
        self,
        merged_df: pd.DataFrame,
        commodities: List[str]
    ) -> Dict[str, Any]:
        """Calculate global correlation across all data."""
        # Create pivot table: each row is a unique (city, date) combination
        # Each column is a commodity price
        pivot_df = merged_df.pivot_table(
            index=['City', 'Date'],
            columns='Commodity',
            values='Price',
            aggfunc='mean'  # Handle any duplicates
        ).reset_index()
        
        # Get only commodity columns for correlation
        commodity_cols = [col for col in pivot_df.columns if col in commodities]
        correlation_df = pivot_df[commodity_cols]
        
        # Calculate correlation matrix
        corr_matrix = correlation_df.corr()
        
        # Calculate p-values
        pvalue_matrix = self._calculate_pvalue_matrix(correlation_df)
        
        # Convert to lists for JSON serialization
        matrix = corr_matrix.round(2).values.tolist()
        pvalues = pvalue_matrix.round(3).values.tolist()
        
        return {
            "mode": "global",
            "commodities": commodities,
            "matrix": matrix,
            "pValues": pvalues,
            "method": "pearson",
            "description": "Correlation matrix showing price relationships between commodities"
        }
    
    def _calculate_per_cluster_correlation(
        self,
        merged_df: pd.DataFrame,
        commodities: List[str]
    ) -> Dict[str, Any]:
        """Calculate correlation per cluster."""
        if 'Cluster' not in merged_df.columns:
            logger.warning("Cluster column not found for per-cluster correlation")
            return {}
        
        clusters = sorted(merged_df['Cluster'].unique())
        cluster_matrices = {}
        
        for cluster in clusters:
            cluster_data = merged_df[merged_df['Cluster'] == cluster]
            
            # Create pivot table for this cluster
            pivot_df = cluster_data.pivot_table(
                index=['City', 'Date'],
                columns='Commodity',
                values='Price',
                aggfunc='mean'
            ).reset_index()
            
            # Get only commodity columns
            commodity_cols = [col for col in pivot_df.columns if col in commodities]
            correlation_df = pivot_df[commodity_cols]
            
            # Check if we have enough data
            if len(correlation_df) < 3:
                logger.warning(f"Insufficient data for cluster {cluster} correlation")
                cluster_matrices[str(cluster)] = {
                    "matrix": [[None] * len(commodities) for _ in range(len(commodities))],
                    "pValues": [[None] * len(commodities) for _ in range(len(commodities))]
                }
                continue
            
            # Calculate correlation and p-values
            corr_matrix = correlation_df.corr()
            pvalue_matrix = self._calculate_pvalue_matrix(correlation_df)
            
            cluster_matrices[str(cluster)] = {
                "matrix": corr_matrix.round(2).values.tolist(),
                "pValues": pvalue_matrix.round(3).values.tolist()
            }
        
        return {
            "mode": "per_cluster",
            "commodities": commodities,
            "clusterMatrices": cluster_matrices,
            "method": "pearson",
            "description": "Correlation matrix showing price relationships per cluster"
        }
    
    def _calculate_pvalue_matrix(self, correlation_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate p-values for correlation matrix."""
        commodities = correlation_df.columns
        n_commodities = len(commodities)
        
        # Initialize p-value matrix
        pvalue_matrix = pd.DataFrame(
            np.ones((n_commodities, n_commodities)),
            index=commodities,
            columns=commodities
        )
        
        # Calculate p-values for each pair
        for i, commodity1 in enumerate(commodities):
            for j, commodity2 in enumerate(commodities):
                if i != j:  # Skip diagonal (self-correlation)
                    # Get valid data points (non-null for both commodities)
                    valid_data = correlation_df[[commodity1, commodity2]].dropna()
                    
                    if len(valid_data) >= 3:  # Minimum for correlation
                        try:
                            _, p_value = pearsonr(valid_data[commodity1], valid_data[commodity2])
                            pvalue_matrix.loc[commodity1, commodity2] = p_value
                        except Exception as e:
                            logger.warning(f"Error calculating p-value for {commodity1}-{commodity2}: {e}")
                            pvalue_matrix.loc[commodity1, commodity2] = None
                    else:
                        pvalue_matrix.loc[commodity1, commodity2] = None
        
        return pvalue_matrix

    def format_pca_data(
        self,
        scaled_features: pd.DataFrame,
        labels: np.ndarray,
        cities: List[str]
    ) -> Dict[str, Any]:
        """
        Calculates 2D coordinates and metadata for scatter plot visualization.

        It uses Principal Component Analysis (PCA) for data with > 2 features, 
        but directly uses the two original features if n_features == 2. It returns 
        an empty dictionary if the input data has fewer than 2 features.

        Args:
            scaled_features: Scaled feature matrix used for clustering (pandas DataFrame).
            labels: Cluster assignment labels (numpy array).
            cities: List of city names (list of strings).

        Returns:
            Dictionary with 2D coordinates and metadata, or an empty dictionary if 
            data is invalid or calculation fails.
        """
        if scaled_features is None or scaled_features.empty:
            logger.warning("Scaled features DataFrame is empty for visualization calculation.")
            return {}
        
        # 1. Prepare Features
        # Use .copy() to prevent SettingWithCopyWarning if scaled_features is a view
        features = scaled_features.drop("City", axis=1).copy() if "City" in scaled_features.columns else scaled_features.copy()
        
        # 2. Check Data Consistency
        if len(features) != len(labels) or len(features) != len(cities):
            logger.warning("Mismatch in data dimensions for visualization calculation (features, labels, or cities).")
            return {}

        n_features = features.shape[1]
        
        # 3. Dimensionality Check and Strategy Selection
        pca_data = {}
        
        if n_features < 2:
            # Strategy A: Insufficient data for 2D plot (Graceful Exit)
            logger.warning(
                f"Skipping 2D visualization: Input data has only {n_features} feature(s), "
                f"but requires at least 2 for a 2-dimensional plot."
            )
            return {}
            
        elif n_features == 2:
            # Strategy B: Use original features directly (Optimized for 2D data)
            logger.info("Using original 2 features for 2D scatter plot (PCA skipped).")
            try:
                # Data must be converted to numpy array for iteration
                transformed_result = features.values 
                feature_names = features.columns.tolist()
                
                x_feature = str(feature_names[0]).split("_")[0]
                y_feature = str(feature_names[1]).split("_")[0]
                
                pca_data = {
                    "components": { # Define components structure for consistency
                        "pc1": {"explained_variance_ratio": 1.0, "explained_variance": 1.0}, # Placeholder, as no variance is calculated
                        "pc2": {"explained_variance_ratio": 1.0, "explained_variance": 1.0},
                    },
                    "feature_contributions": { # Label axes with original feature names
                        "pc1": {name: 1.0 if i == 0 else 0.0 for i, name in enumerate(feature_names)},
                        "pc2": {name: 1.0 if i == 1 else 0.0 for i, name in enumerate(feature_names)},
                    },
                    "method": "Original Features",
                    "description": f"Direct plot of features: {x_feature} vs. {y_feature}",
                    "features": {
                        "x_axis": x_feature,
                        "y_axis": y_feature
                    },
                }
            except Exception as e:
                logger.error(f"Error preparing 2-feature data for plotting: {e}")
                return {}
            
        else: # n_features > 2
            # Strategy C: Apply PCA (Intended for high-dimensional data)
            try:
                pca = PCA(n_components=2, random_state=42)
                transformed_result = pca.fit_transform(features)
                feature_names = features.columns.tolist()

                pca_data = {
                    "components": {
                        "pc1": {
                            "explained_variance_ratio": float(pca.explained_variance_ratio_[0]),
                            "explained_variance": float(pca.explained_variance_[0])
                        },
                        "pc2": {
                            "explained_variance_ratio": float(pca.explained_variance_ratio_[1]),
                            "explained_variance": float(pca.explained_variance_[1])
                        }
                    },
                    "feature_contributions": {
                        "pc1": {name: float(pca.components_[0][i]) for i, name in enumerate(feature_names)},
                        "pc2": {name: float(pca.components_[1][i]) for i, name in enumerate(feature_names)}
                    },
                    "method": "PCA",
                    "description": "Principal Component Analysis (PCA) reduction to 2 dimensions."
                }
            except Exception as e:
                logger.error(f"Error calculating PCA data: {e}")
                return {}

        # 4. Create transformed data points (common to both Strategy B and C)
        transformed_data = [
            {
                "x": float(x),
                "y": float(y),
                "clusterId": int(labels[i]),
                "cityName": cities[i],
                "originalIndex": i
            } 
            for i, (x, y) in enumerate(transformed_result)
        ]
        
        # 5. Finalize and Return
        pca_data["transformed_data"] = transformed_data
        logger.info(f"Generated 2D visualization data for {len(transformed_data)} cities using {pca_data['method']}.")
        return pca_data

    def format_silhouette_data(
        self,
        scaled_features: pd.DataFrame,
        labels: np.ndarray,
        cities: List[str]
    ) -> Dict[str, Any]:
        """
        Calculate silhouette scores for bar chart visualization.
        
        Args:
            scaled_features: Scaled feature matrix used for clustering
            labels: Cluster assignment labels
            cities: List of city names
            
        Returns:
            Dictionary with silhouette data for bar chart
        """
        if scaled_features is None or scaled_features.empty:
            return {}
        
        if len(scaled_features) != len(labels) or len(scaled_features) != len(cities):
            logger.warning("Mismatch in data dimensions for silhouette calculation")
            return {}
        
        try:
            # Remove City column if present (for feature matrix)
            feature_matrix = scaled_features.copy()
            if "City" in feature_matrix.columns:
                feature_matrix = feature_matrix.drop("City", axis=1)
            
            # Calculate overall clustering metrics
            overall_silhouette = silhouette_score(feature_matrix, labels)
            davies_bouldin = davies_bouldin_score(feature_matrix, labels)
            
            # Calculate individual silhouette scores
            individual_scores = silhouette_samples(feature_matrix, labels)
            
            # Create city silhouette data with sorting
            city_silhouettes = []
            for i, (city, score, cluster_id) in enumerate(zip(cities, individual_scores, labels)):
                city_silhouettes.append({
                    "city": city,
                    "silhouette": float(score),
                    "clusterId": int(cluster_id)
                })
            
            # Sort by silhouette score (highest first)
            city_silhouettes.sort(key=lambda x: x["silhouette"], reverse=True)
            
            silhouette_data = {
                "clusteringMetrics": {
                    "overall_silhouette": float(overall_silhouette),
                    "davies_bouldin": float(davies_bouldin)
                },
                "citySilhouettes": city_silhouettes
            }
            
            logger.info(f"Calculated silhouette data for {len(city_silhouettes)} cities")
            return silhouette_data
            
        except Exception as e:
            logger.error(f"Error calculating silhouette data: {e}")
            return {}

    def format_frontend_response(
        self,
        analysis_id: str,
        labels: np.ndarray,
        cities: List[str],
        preprocessed_df: Optional[pd.DataFrame] = None,
        merged_df: Optional[pd.DataFrame] = None,
        commodities: Optional[List[str]] = None,
        years: Optional[List[int]] = None,
        clusters_palette: Optional[List[Dict[str, Any]]] = None,
        config: Optional[Any] = None,
        scaled_features: Optional[pd.DataFrame] = None
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

        # Radar features (optional if merged_df provided)
        radar_features: Dict[str, Any] = {}
        if merged_df is not None and not merged_df.empty:
            radar_features = self.format_radar_features(
                merged_df=merged_df,
                labels=labels,
                cities=cities
            )

        # Boxplot data (optional if merged_df provided)
        boxplot_data: Dict[str, Any] = {}
        if merged_df is not None and not merged_df.empty:
            boxplot_data = self.format_boxplot_data(
                merged_df=merged_df,
                labels=labels,
                cities=cities
            )

        # Correlation matrix (optional if merged_df provided)
        correlation_matrix: Dict[str, Any] = {}
        if merged_df is not None and not merged_df.empty:
            correlation_matrix = self.format_correlation_matrix(
                merged_df=merged_df,
                config=config
            )

        # PCA data (optional if scaled features provided)
        pca_data: Dict[str, Any] = {}
        if scaled_features is not None and not scaled_features.empty:
            pca_data = self.format_pca_data(
                scaled_features=scaled_features,
                labels=labels,
                cities=cities
            )

        # Silhouette data (optional if scaled features provided)
        silhouette_data: Dict[str, Any] = {}
        if scaled_features is not None and not scaled_features.empty:
            silhouette_data = self.format_silhouette_data(
                scaled_features=scaled_features,
                labels=labels,
                cities=cities
            )

        response = {
            "analysis_id": analysis_id,
            "years": years_list,
            "clusters": clusters_palette,
            "cities": assignments,
            "trends": trends
        }
        
        # Only add radarFeatures if we have data
        if radar_features:
            response["radarFeatures"] = radar_features
        
        # Only add boxPlotData if we have data
        if boxplot_data:
            response["boxPlotData"] = boxplot_data
        
        # Only add correlationMatrix if we have data
        if correlation_matrix:
            response["correlationMatrix"] = correlation_matrix
        
        # Only add pcaData if we have data
        if pca_data:
            response["pcaData"] = pca_data
        
        # Only add silhouette data if we have data
        if silhouette_data:
            response.update(silhouette_data)  # Add clusteringMetrics and citySilhouettes directly

        return response
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
