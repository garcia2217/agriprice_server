import sys
import os
import logging
import time
import warnings
import json
import folium
from IPython.display import display
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Tuple

# Configure matplotlib to use non-interactive backend to avoid tkinter issues
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from yellowbrick.cluster import SilhouetteVisualizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats

# Scikit-learn imports
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, silhouette_samples

# PDF generation imports
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch

# Selenium for map conversion
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# UMAP for visualization (optional)
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    print("⚠️ Warning: umap-learn not available. Will use PCA for 2D scatter plot.")
    UMAP_AVAILABLE = False

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Assume ROOT_DIR and SAVE_DIR are set up correctly
ROOT_DIR = Path.cwd()
SAVE_DIR = ROOT_DIR / "clustering_results"
SAVE_DIR.mkdir(parents=True, exist_ok=True)


class ClusterVisualizationService:
    """
    A service dedicated to generating visualizations from pre-computed clustering results.
    """
    def __init__(
        self,
        scaled_df: pd.DataFrame,
        preprocessed_df: pd.DataFrame,
        labels: np.ndarray,
        model: Any, # The fitted clustering model object
        silhouette_avg: float,
        algorithm_name: str, # e.g., "kmeans", "spectral"
        output_path: Path
    ):
        """
        Initializes the visualization service with clustering results.

        Args:
            scaled_df (pd.DataFrame): DataFrame with scaled features used for clustering.
            preprocessed_df (pd.DataFrame): Original preprocessed data for price analysis.
            labels (np.ndarray): The cluster labels assigned to each data point.
            model (Any): The fitted model from scikit-learn (e.g., KMeans instance).
            silhouette_avg (float): The average silhouette score for the clustering.
            algorithm_name (str): The name of the algorithm used (for titling).
            output_path (Path): The directory path to save all visualization files.
        """
        self.scaled_df = scaled_df
        self.X = scaled_df.drop(columns=["City"]) # Feature matrix
        self.preprocessed_df = preprocessed_df
        self.labels = labels
        self.model = model
        self.silhouette_avg = silhouette_avg
        self.algorithm_name = algorithm_name
        self.output_path = output_path
        self.output_path.mkdir(parents=True, exist_ok=True)
        print(f"--- Visualization Service Initialized for '{algorithm_name}' ---")
        print(f"Output will be saved to: {self.output_path}")

    def generate_all_visualizations(self, temp_dir: Path = None) -> Dict[str, Path]:
        """
        Runs all visualization generation methods in sequence.
        Returns dictionary of chart types to file paths.
        """
        if temp_dir is None:
            temp_dir = self.output_path
            
        print("\n🚀 Starting visualization generation...")
        
        # Prepare common data structures
        cluster_map = pd.DataFrame({"City": self.scaled_df["City"], "Cluster": self.labels})
        merged_df = self.preprocessed_df.merge(cluster_map, on="City", how="left")

        # Dictionary to store image paths
        image_paths = {}
        
        # 1. Generate Silhouette Visualization
        silhouette_path = temp_dir / "silhouette_plot.png"
        self._generate_silhouette_visualization(silhouette_path)
        image_paths['silhouette'] = silhouette_path
        
        # 2. Generate 2D Scatter Plot
        scatter_path = temp_dir / "scatter_plot.png"
        self._generate_scatter_plot(scatter_path)
        image_paths['scatter'] = scatter_path
        
        # 3. Generate Box Plot for Price Distribution
        boxplot_path = temp_dir / "boxplot.png"
        self._generate_box_plot(merged_df, boxplot_path)
        image_paths['boxplot'] = boxplot_path
        
        # 4. Generate Time-Series Line Chart
        linechart_path = temp_dir / "linechart.png"
        self._generate_line_chart(merged_df, linechart_path)
        image_paths['linechart'] = linechart_path
        
        # 5. Generate Radar Chart
        df_radar_data = self._prepare_data_for_radar()
        radar_path = temp_dir / "radar.png"
        self._plot_radar_chart(df_radar_data, radar_path)
        image_paths['radar'] = radar_path
        
        # 6. Generate Map Visualization
        coordinates_file = Path("data/city_coordinates.json")
        df_for_map = self._prepare_data_for_map(cluster_map, coordinates_file)
        map_path = temp_dir / "map.png"
        
        # Try Folium map first, fallback to static map if it fails
        try:
            self._create_cluster_map_png(df_for_map, map_path)
            if not map_path.exists():
                print("⚠️ Folium map failed, using static map fallback")
                self._create_static_map(df_for_map, map_path)
        except Exception as e:
            print(f"⚠️ Folium map error: {e}, using static map fallback")
            self._create_static_map(df_for_map, map_path)
        
        image_paths['map'] = map_path
        
        # 7. Generate Cluster Members Summary (optional for PDF)
        self._generate_cluster_summary(cluster_map)
        
        print("\n✅ All visualizations have been generated successfully!")
        return image_paths

    def _generate_silhouette_visualization(self, output_path: Path):
        print("  - Generating Silhouette Plot...")
        # Yellowbrick visualizer works best for KMeans and some other scikit-learn models
        if self.algorithm_name in ["kmeans"] and hasattr(self.model, 'cluster_centers_'):
            visualizer = SilhouetteVisualizer(self.model, colors='yellowbrick')
            visualizer.fit(self.X)
            visualizer.show(outpath=str(output_path))
            plt.close()
        else:
            # Generic fallback for other algorithms like Spectral or FCM results
            fig, ax = plt.subplots(1, 1)
            fig.set_size_inches(10, 7)
            n_clusters = len(np.unique(self.labels))
            ax.set_ylim([0, len(self.X) + (n_clusters + 1) * 10])
            sample_silhouette_values = silhouette_samples(self.X, self.labels)
            y_lower = 10
            for i in range(n_clusters):
                ith_cluster_silhouette_values = sample_silhouette_values[self.labels == i]
                ith_cluster_silhouette_values.sort()
                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i
                color = cm.nipy_spectral(float(i) / n_clusters)
                ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                                  facecolor=color, edgecolor=color, alpha=0.7)
                ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
                y_lower = y_upper + 10
            ax.axvline(x=self.silhouette_avg, color="red", linestyle="--")
            ax.set_title(f"Silhouette Plot for {self.algorithm_name.title()} Clustering")
            ax.set_xlabel("Silhouette coefficient values")
            ax.set_ylabel("Cluster label")
            ax.set_yticks([])
            ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        print("    ✓ Done.")

    def _generate_scatter_plot(self, output_path: Path):
        pca = PCA(n_components=2, random_state=42)
        X_reduced = pca.fit_transform(self.X)
        title = "PCA 2D Cluster Visualization"

        plt.figure(figsize=(10, 8))
        n_clusters = len(np.unique(self.labels))
        palette = sns.color_palette("tab10", n_colors=n_clusters)
        sns.scatterplot(
            x=X_reduced[:, 0],
            y=X_reduced[:, 1],
            hue=self.labels,
            palette=palette,
            s=80,
            alpha=0.9,
            edgecolor='k'
        )
        plt.title(title, fontsize=16)
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.legend(title="Cluster")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print("    ✓ Done.")

    def _generate_box_plot(self, merged_df, output_path: Path):
        # This method remains largely the same as it already consumes processed data
        print("  - Generating Price Distribution Box Plots...")
        
        if 'Year' not in merged_df.columns:
            merged_df['Year'] = pd.to_datetime(merged_df['Date']).dt.year

        commodities = sorted(merged_df["Commodity"].unique())
        clusters = sorted(merged_df["Cluster"].unique())
        n_commodities = len(commodities)
        n_clusters = len(clusters)

        print(f"📊 Plotting {n_commodities} commodities across {n_clusters} clusters")
        print(f"Years in dataset: {sorted(merged_df['Year'].unique())}")

        n_cols = 2
        n_rows = int(np.ceil(n_commodities / n_cols))

        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 5))
        axes = axes.flatten() if n_commodities > 1 else [axes]

        palette = sns.color_palette("Set2", n_colors=n_clusters)
        cluster_colors = dict(zip(clusters, palette))

        for i, commodity in enumerate(commodities):
            ax = axes[i]
            
            df_commodity = merged_df[merged_df['Commodity'] == commodity]
            
            # Check if there's data
            if df_commodity.empty:
                ax.text(0.5, 0.5, 'No Data Available', 
                    ha='center', va='center', fontsize=12)
                ax.set_title(commodity, fontsize=14, weight='bold')
                continue
            
            # Create boxplot with consistent cluster ordering
            sns.boxplot(
                x='Year', 
                y='Price', 
                hue='Cluster',
                hue_order=clusters,
                data=df_commodity,
                ax=ax,
                palette=cluster_colors,
                fliersize=3,
                linewidth=1.3,
                showmeans=True,
                meanprops=dict(marker='D', markerfacecolor='white', 
                            markeredgecolor='black', markersize=5)
            )
            
            ax.set_title(f"{commodity}", fontsize=14, weight='bold', pad=12)
            ax.set_xlabel("Year", fontsize=11, weight='semibold')
            ax.set_ylabel("Price (Rp)", fontsize=11, weight='semibold')
            ax.grid(True, axis='y', linestyle='--', alpha=0.3)
            
            # Remove individual legends (we'll add one global legend)
            if ax.get_legend():
                ax.get_legend().remove()
            
            # Rotate x-axis labels if there are many years
            if len(df_commodity['Year'].unique()) > 3:
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Remove empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        # Create a single legend for the entire figure
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            ncol = min(n_clusters, 6)
            fig.legend(
                handles, labels,
                title='City Cluster',
                loc='upper center',
                bbox_to_anchor=(0.5, 0.99),
                ncol=ncol,
                fontsize=11,
                title_fontsize=12,
                frameon=True,
                fancybox=True,
                shadow=True,
                edgecolor='gray'
            )

        # Add title
        year_min = merged_df['Year'].min()
        year_max = merged_df['Year'].max()
        fig.suptitle(
            f"Distribusi Harga per Komoditas berdasarkan Tahun dan Klaster ({year_min}–{year_max})", 
            fontsize=18, 
            weight='bold', 
            y=1.01
        )

        plt.tight_layout()

        # Save the figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Print color mapping
        print("\n🎨 Cluster Color Legend:")
        print("-" * 40)
        for cluster in clusters:
            color = cluster_colors[cluster]
            hex_color = '#{:02x}{:02x}{:02x}'.format(
                int(color[0]*255), int(color[1]*255), int(color[2]*255)
            )
            print(f"■ {cluster}: {hex_color}")

        print(f"\n✓ Plot saved as '{output_path.name}'")
        print(f"✓ Total data points: {len(merged_df)}")
        print("    ✓ Done.")

    def _generate_line_chart(self, merged_df, output_path: Path):
        try:
            print("  - Generating Price Trend Line Charts...")
            
            # Ensure Date column is datetime
            if 'Date' not in merged_df.columns or merged_df['Date'].dtype != 'datetime64[ns]':
                merged_df['Date'] = pd.to_datetime(merged_df['Date'])
            
            print("=== Data Information ===")
            print(f"Date range: {merged_df['Date'].min()} to {merged_df['Date'].max()}")
            print(f"Years: {sorted(merged_df['Date'].dt.year.unique())}")
            print(f"Clusters: {sorted(merged_df['Cluster'].unique())}")
            print(f"Commodities: {sorted(merged_df['Commodity'].unique())}\n")

            # --- 2. Process Data for Plotting ---
            trend = (
                merged_df
                .groupby(["Date", "Cluster", "Commodity"], as_index=False)["Price"]
                .mean()
                .sort_values("Date")
            )
            
            # Ensure Date is datetime (redundant but safe)
            trend["Date"] = pd.to_datetime(trend["Date"])

            # --- 3. Setup for Visualization ---
            commodities = sorted(trend["Commodity"].unique())
            clusters = sorted(trend["Cluster"].unique())
            n_commodities = len(commodities)
            n_clusters = len(clusters)
            
            # Create consistent color palette
            palette = dict(zip(clusters, sns.color_palette("tab10", n_colors=n_clusters)))
            
            # Calculate date limits from actual data
            min_date_limit = trend['Date'].min()
            max_date_limit = trend['Date'].max()
            
            print(f"Plotting {n_commodities} commodities with {n_clusters} clusters")
            print(f"Plot date range: {min_date_limit.date()} to {max_date_limit.date()}\n")

            # --- 4. Create Subplots ---
            n_cols = 2
            n_rows = int(np.ceil(n_commodities / n_cols))
            fig, axes = plt.subplots(
                n_rows, n_cols, 
                figsize=(17, n_rows * 4.5), 
                constrained_layout=True
            )
            axes = axes.flatten() if n_commodities > 1 else [axes]

            for i, commodity in enumerate(commodities):
                ax = axes[i]
                df_c = trend[trend["Commodity"] == commodity]
                
                # Check for empty data
                if df_c.empty:
                    ax.text(0.5, 0.5, 'No Data Available', 
                        ha='center', va='center', fontsize=12)
                    ax.set_title(commodity, fontsize=15, weight='bold')
                    continue

                # Plot with consistent cluster ordering
                sns.lineplot(
                    data=df_c,
                    x="Date",
                    y="Price",
                    hue="Cluster",
                    hue_order=clusters,  # Ensure consistent order
                    ax=ax,
                    palette=palette,
                    linewidth=2,
                    alpha=0.85,
                    legend=False  # We'll add custom legend
                )

                ax.set_title(commodity, fontsize=15, weight='bold', pad=10)
                ax.set_ylabel("Average Price (Rp)", fontsize=11)
                ax.set_xlabel("")
                ax.grid(True, linestyle='--', alpha=0.4, linewidth=0.8)
                
                # Set x-axis to show only years within data range
                ax.xaxis.set_major_locator(mdates.YearLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                
                # CRITICAL: Set x-axis limits to prevent empty years
                ax.set_xlim(min_date_limit, max_date_limit)
                
                # Style x-axis
                ax.tick_params(axis='x', rotation=0, labelsize=10)
                
                # Add custom legend to each subplot
                handles = [plt.Line2D([0], [0], color=palette[cluster], 
                                    linewidth=2, label=cluster) 
                        for cluster in clusters]
                ax.legend(
                    handles=handles,
                    title="Cluster",
                    loc='best',
                    fontsize=9,
                    title_fontsize=10,
                    frameon=True,
                    framealpha=0.9,
                    edgecolor='gray',
                    fancybox=True
                )

            # --- 5. Clean Up ---
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])
            
            # Add main title
            year_min = min_date_limit.year
            year_max = max_date_limit.year
            fig.suptitle(
                f"Tren Rata-rata Harga Pangan per Klaster ({year_min}–{year_max})",
                fontsize=19,
                weight='bold',
                y=1.04
            )
            
            # Save and display
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Plot saved as '{output_path.name}'")
            print(f"✓ Successfully plotted {n_commodities} commodities")
            
            # Print color mapping
            print("\n🎨 Cluster Color Legend:")
            print("-" * 40)
            for cluster in clusters:
                color = palette[cluster]
                hex_color = '#{:02x}{:02x}{:02x}'.format(
                    int(color[0]*255), int(color[1]*255), int(color[2]*255)
                )
                print(f"■ {cluster}: {hex_color}")
            
            print("    ✓ Done.")

        except KeyError as e:
            print(f"❌ Error: Missing column - {e}")
            print(f"Available columns: {merged_df.columns.tolist()}")
        except Exception as e:
            print(f"❌ An error occurred: {e}")
            import traceback
            traceback.print_exc()

    def _prepare_data_for_radar(self) -> pd.DataFrame:
        """
        Prepares data for the radar chart by calculating the mean scaled value
        for each commodity within each cluster.
        """
        print("  - Preparing data for Radar Chart...")
        df_with_labels = self.X.copy()
        df_with_labels['Cluster'] = self.labels
        
        # Group by cluster and find the mean value for each feature (commodity)
        radar_data = df_with_labels.groupby('Cluster').mean()
        
        print("    ✓ Radar data prepared.")
        return radar_data

    def _plot_radar_chart(self, df_radar: pd.DataFrame, output_path: Path):
        """
        Generates a radar chart where axes are commodities.
        """
        print("  - Generating Radar Chart...")
        labels = df_radar.columns
        num_vars = len(labels)
        
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
        
        n_clusters = len(df_radar)
        colors = plt.cm.get_cmap('tab10', n_clusters)

        for i, (cluster_id, row) in enumerate(df_radar.iterrows()):
            values = row.tolist()
            values += values[:1]
            ax.plot(angles, values, color=colors(i), linewidth=2, linestyle='solid', label=f'Cluster {cluster_id}')
            ax.fill(angles, values, color=colors(i), alpha=0.25)

        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, size=12)
        
        plt.title('Cluster Profile by Average Scaled Commodity Value', size=20, color='black', y=1.1)
        plt.legend(title='Clusters', loc='upper right', bbox_to_anchor=(1.4, 1.1))
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print("    ✓ Done.")

    def _prepare_data_for_map(self, cluster_data: pd.DataFrame, coordinates_filepath: Path) -> pd.DataFrame:
        """
        Consolidates city, cluster, and coordinate data into a single DataFrame.

        Args:
            cluster_data: DataFrame from clustering containing 'City' and 'Cluster' columns.
            coordinates_filepath: Path to the user-provided JSON file with city coordinates.

        Returns:
            A DataFrame ready for plotting, with columns for City, Cluster, Lat, and Lon.
        """
        try:
            # --- 1. Load the coordinates mapping from your JSON file ---
            with open(coordinates_filepath, 'r') as f:
                city_coordinates = json.load(f)
            
            # Convert it into a DataFrame
            df_coords = pd.DataFrame(city_coordinates.items(), columns=['City', 'Coordinates'])
            df_coords[['Lat', 'Lon']] = pd.DataFrame(df_coords['Coordinates'].tolist(), index=df_coords.index)
            df_coords.drop('Coordinates', axis=1, inplace=True)
            
            # --- 2. Get the city-to-cluster mapping ---
            df_city_clusters = cluster_data[['City', 'Cluster']].drop_duplicates().reset_index(drop=True)

            # --- 3. Merge the DataFrames ---
            df_map_data = pd.merge(df_city_clusters, df_coords, on='City', how='inner')
            
            print("Data successfully prepared for mapping.")
            print(f"Found coordinates for {len(df_map_data)} cities.")
            
            return df_map_data

        except FileNotFoundError:
            print(f"Error: The file '{coordinates_filepath}' was not found.")
            return pd.DataFrame()
        except Exception as e:
            print(f"An error occurred during data preparation: {e}")

    def _create_cluster_map(self, df_map_data: pd.DataFrame):
        """
        Creates and displays an interactive map visualizing the city clusters.

        Args:
            df_map_data: DataFrame from Part 1 containing City, Cluster, Lat, and Lon.
            output_filename: The name of the HTML file to save the map to.
            
        Returns:
            A folium.Map object that will be displayed in the notebook.
        """
        if df_map_data.empty:
            print("Input data for map is empty. Cannot generate visualization.")
            return None

        # Validate required columns
        required_cols = ['City', 'Cluster', 'Lat', 'Lon']
        missing_cols = [col for col in required_cols if col not in df_map_data.columns]
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            return None

        # --- 1. Initialize the map centered on Indonesia ---
        map_center = [-2.548926, 118.0148634]
        cluster_map = folium.Map(
            location=map_center, 
            zoom_start=5, 
            tiles="OpenStreetMap"
        )

        # --- 2. Create a color palette for the clusters ---
        num_clusters = int(df_map_data['Cluster'].max() + 1)
        # Using a built-in colormap for clear, distinct colors
        from matplotlib.colors import to_hex
        colors = plt.cm.get_cmap('tab10', num_clusters)
        cluster_colors = {i: to_hex(colors(i)) for i in range(num_clusters)}
        
        print(f"🗺️  Creating map with {num_clusters} clusters")
        print(f"📍 Plotting {len(df_map_data)} cities")

        # --- 3. Add a marker for each city ---
        for _, row in df_map_data.iterrows():
            cluster_id = int(row['Cluster'])
            folium.CircleMarker(
                location=[row['Lat'], row['Lon']],
                radius=8,  # Slightly larger for better visibility
                color=cluster_colors.get(cluster_id, '#808080'),
                fill=True,
                fill_color=cluster_colors.get(cluster_id, '#808080'),
                fill_opacity=0.7,
                weight=2,  # Border thickness
                popup=folium.Popup(
                    f"<b>{row['City']}</b><br>Cluster: {cluster_id}", 
                    max_width=200
                )
            ).add_to(cluster_map)

        # --- 4. Add a legend ---
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; right: 50px; width: 180px; height: auto; 
                    background-color: white; z-index:9999; font-size:14px;
                    border:2px solid grey; border-radius: 5px; padding: 10px">
        <p style="margin-bottom: 5px; font-weight: bold;">City Clusters</p>
        '''
        for cluster_id, color in cluster_colors.items():
            legend_html += f'''
            <p style="margin: 3px 0;">
                <span style="background-color:{color}; 
                            width: 20px; height: 20px; 
                            display: inline-block; 
                            border: 1px solid black;
                            margin-right: 5px;"></span>
                Cluster {cluster_id}
            </p>
            '''
        legend_html += '</div>'
        cluster_map.get_root().html.add_child(folium.Element(legend_html))

        # --- 5. Save the map to an HTML file ---
        save_path = self.output_path / "cluster_map.html"
        try:
            cluster_map.save(str(save_path))
            print(f"✓ Map saved as cluster_map.html")
        except Exception as e:
            print(f"❌ Error saving map: {e}")
            
        # Print cluster color mapping
        print("\n🎨 Cluster Colors:")
        for cluster_id, color in cluster_colors.items():
            print(f"  Cluster {cluster_id}: {color}")
        
        return cluster_map

    def _create_cluster_map_png(self, df_map_data: pd.DataFrame, output_path: Path):
        """
        Creates cluster map and saves as PNG using Selenium.
        """
        print("  - Generating Map Visualization...")
        
        if df_map_data.empty:
            print("Input data for map is empty. Cannot generate visualization.")
            return None

        # Validate required columns
        required_cols = ['City', 'Cluster', 'Lat', 'Lon']
        missing_cols = [col for col in required_cols if col not in df_map_data.columns]
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            return None

        # Create folium map
        map_center = [-2.548926, 118.0148634]
        cluster_map = folium.Map(
            location=map_center, 
            zoom_start=5, 
            tiles="OpenStreetMap"
        )

        # Create color palette
        num_clusters = int(df_map_data['Cluster'].max() + 1)
        from matplotlib.colors import to_hex
        colors = plt.cm.get_cmap('tab10', num_clusters)
        cluster_colors = {i: to_hex(colors(i)) for i in range(num_clusters)}
        
        print(f"🗺️  Creating map with {num_clusters} clusters")
        print(f"📍 Plotting {len(df_map_data)} cities")

        # Add markers
        for _, row in df_map_data.iterrows():
            cluster_id = int(row['Cluster'])
            folium.CircleMarker(
                location=[row['Lat'], row['Lon']],
                radius=8,
                color=cluster_colors.get(cluster_id, '#808080'),
                fill=True,
                fill_color=cluster_colors.get(cluster_id, '#808080'),
                fill_opacity=0.7,
                weight=2,
                popup=folium.Popup(
                    f"<b>{row['City']}</b><br>Cluster: {cluster_id}", 
                    max_width=200
                )
            ).add_to(cluster_map)

        # Add legend
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; right: 50px; width: 180px; height: auto; 
                    background-color: white; z-index:9999; font-size:14px;
                    border:2px solid grey; border-radius: 5px; padding: 10px">
        <p style="margin-bottom: 5px; font-weight: bold;">City Clusters</p>
        '''
        for cluster_id, color in cluster_colors.items():
            legend_html += f'''
            <p style="margin: 3px 0;">
                <span style="background-color:{color}; 
                            width: 20px; height: 20px; 
                            display: inline-block; 
                            border: 1px solid black;
                            margin-right: 5px;"></span>
                Cluster {cluster_id}
            </p>
            '''
        legend_html += '</div>'
        cluster_map.get_root().html.add_child(folium.Element(legend_html))

        # Save as HTML first
        temp_html_path = output_path.parent / "temp_map.html"
        cluster_map.save(str(temp_html_path))
        
        try:
            # Convert HTML to PNG using Selenium
            options = Options()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-gpu')
            options.add_argument('--window-size=1200,800')
            options.add_argument('--disable-web-security')
            options.add_argument('--disable-features=VizDisplayCompositor')
            options.add_argument('--disable-extensions')
            options.add_argument('--disable-plugins')
            options.add_argument('--disable-images')
            options.add_argument('--disable-javascript')
            
            driver = webdriver.Chrome(options=options)
            driver.set_page_load_timeout(30)
            driver.get(f'file://{temp_html_path.absolute()}')
            
            # Wait for map to load
            import time
            time.sleep(5)  # Increased wait time for map tiles to load
            
            # Take screenshot
            driver.save_screenshot(str(output_path))
            driver.quit()
            
            # Clean up HTML file
            temp_html_path.unlink()
            
            print(f"✓ Map saved as '{output_path.name}'")
            print("    ✓ Done.")
            
        except Exception as e:
            print(f"❌ Error converting map to PNG: {e}")
            # Clean up HTML file
            if temp_html_path.exists():
                temp_html_path.unlink()
            return None

    def _create_static_map(self, df_map_data: pd.DataFrame, output_path: Path):
        """
        Create a static map using matplotlib instead of Folium (fallback method).
        """
        print("  - Generating Static Map Visualization...")
        
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Circle
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create color palette
            num_clusters = int(df_map_data['Cluster'].max() + 1)
            colors = plt.cm.get_cmap('tab10', num_clusters)
            cluster_colors = {i: colors(i) for i in range(num_clusters)}
            
            # Plot cities
            for _, row in df_map_data.iterrows():
                cluster_id = int(row['Cluster'])
                color = cluster_colors.get(cluster_id, 'gray')
                
                ax.scatter(row['Lon'], row['Lat'], 
                          c=[color], s=100, alpha=0.7, 
                          edgecolors='black', linewidth=1)
                
                # Add city name
                ax.annotate(row['City'], (row['Lon'], row['Lat']), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.8)
            
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_title('City Clusters Geographic Distribution')
            ax.grid(True, alpha=0.3)
            
            # Add legend
            for cluster_id in range(num_clusters):
                color = cluster_colors[cluster_id]
                ax.scatter([], [], c=[color], s=100, label=f'Cluster {cluster_id}')
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Static map saved as '{output_path.name}'")
            print("    ✓ Done.")
            
        except Exception as e:
            print(f"❌ Error creating static map: {e}")
            return None

    def _generate_cluster_summary(self, cluster_map: pd.DataFrame):
        print("  - Generating cluster summary file...")
        df_cluster_summary = cluster_map.groupby("Cluster")['City'].agg(
            number_of_members='count',
            members=lambda cities: ', '.join(sorted(cities))
        ).reset_index().sort_values("Cluster").set_index("Cluster")
        
        df_cluster_summary.to_excel(self.output_path / "cluster_members_summary.xlsx")
        print("    ✓ Done.")

    def generate_pdf_report(self, output_pdf_path: Path) -> Path:
        """
        Generate PDF report with all visualizations.
        """
        print("\n📄 Generating PDF report...")
        
        # Create temporary directory for plots
        temp_plots_dir = output_pdf_path.parent / "temp_plots"
        temp_plots_dir.mkdir(exist_ok=True)
        
        try:
            # Generate all visualizations
            image_paths = self.generate_all_visualizations(temp_plots_dir)
            
            # Create PDF
            doc = SimpleDocTemplate(str(output_pdf_path), pagesize=A4)
            story = []
            styles = getSampleStyleSheet()
            
            # Title
            title = Paragraph("Clustering Analysis Report", styles['Title'])
            story.append(title)
            story.append(Spacer(12, 12))
            
            # Add metadata
            metadata = f"""
            <b>Algorithm:</b> {self.algorithm_name.title()}<br/>
            <b>Number of Clusters:</b> {len(np.unique(self.labels))}<br/>
            <b>Silhouette Score:</b> {self.silhouette_avg:.4f}<br/>
            <b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            story.append(Paragraph(metadata, styles['Normal']))
            story.append(Spacer(12, 12))
            
            # Add each chart
            charts = [
                ("Silhouette Analysis", image_paths.get('silhouette')),
                ("PCA Scatter Plot", image_paths.get('scatter')),
                ("Price Distribution", image_paths.get('boxplot')),
                ("Price Trends", image_paths.get('linechart')),
                ("Radar Chart", image_paths.get('radar')),
                ("Geographic Distribution", image_paths.get('map'))
            ]
            
            charts_added = 0
            for title_text, image_path in charts:
                if image_path and image_path.exists():
                    try:
                        story.append(Paragraph(title_text, styles['Heading2']))
                        story.append(Image(str(image_path), width=400, height=300))
                        story.append(Spacer(12, 12))
                        charts_added += 1
                    except Exception as e:
                        print(f"⚠️ Warning: Could not add {title_text} to PDF: {e}")
                else:
                    print(f"⚠️ Warning: {title_text} image not found")
            
            if charts_added == 0:
                story.append(Paragraph("No visualizations could be generated.", styles['Normal']))
            
            # Build PDF
            doc.build(story)
            
            # Clean up temporary plots
            import shutil
            shutil.rmtree(temp_plots_dir)
            
            print(f"✓ PDF report saved as '{output_pdf_path.name}' with {charts_added} charts")
            return output_pdf_path
            
        except Exception as e:
            print(f"❌ Error generating PDF: {e}")
            # Clean up on error
            if temp_plots_dir.exists():
                import shutil
                shutil.rmtree(temp_plots_dir)
            raise e

# --- Helper functions for plotting (copy from your original file) ---
# You would copy the full code for `_generate_box_plot`, `_generate_line_chart`,
# `_prepare_data_for_map`, and `_create_cluster_map` into the class above.

def main():
    """
    Example workflow:
    1. Load data.
    2. Run a clustering algorithm to get results.
    3. Pass the results to the visualization service.
    """
    # --- 1. Load Data ---
    SCALED_DATA_PATH = Path("data/master/features/master_food_prices.parquet")
    PREPROCESSED_DATA_PATH = Path("data/master/features/master_food_prices.parquet")
    
    scaled_df = pd.read_csv(SCALED_DATA_PATH)
    preprocessed_df = pd.read_parquet(PREPROCESSED_DATA_PATH)
    
    # Simple cleaning if needed
    if "Unnamed: 0" in scaled_df.columns:
        scaled_df = scaled_df.drop(columns=["Unnamed: 0"])
        
    X = scaled_df.drop(columns=["City"])
    
    # --- 2. Run Clustering (Example with KMeans) ---
    algorithm_to_run = "kmeans"
    n_clusters_to_run = 4

    print(f"Running {algorithm_to_run.upper()} with K={n_clusters_to_run} to get clustering results...")
    
    model = KMeans(
        n_clusters=n_clusters_to_run,
        random_state=42,
        n_init=10
    )
    labels = model.fit_predict(X)
    silhouette_avg = silhouette_score(X, labels)
    print(f"Clustering complete. Silhouette Score: {silhouette_avg:.4f}")

    # --- 3. Run Visualization Service ---
    # Define where to save the output for this specific run
    output_directory = SAVE_DIR / f"{algorithm_to_run}" / f"k{n_clusters_to_run}"
    
    # Instantiate the service with the results
    viz_service = ClusterVisualizationService(
        scaled_df=scaled_df,
        preprocessed_df=preprocessed_df,
        labels=labels,
        model=model,
        silhouette_avg=silhouette_avg,
        algorithm_name=algorithm_to_run,
        output_path=output_directory
    )
    
    # Generate all plots
    viz_service.generate_all_visualizations()


if __name__ == "__main__":
    main()