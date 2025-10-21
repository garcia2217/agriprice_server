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
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps

# Configure matplotlib to use non-interactive backend to avoid tkinter issues
import matplotlib
matplotlib.use('Agg')

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
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader

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


def safe_plot_generation(plot_name: str):
    """Decorator to handle errors in individual plot generation gracefully."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                print(f"  - Generating {plot_name}...")
                result = func(self, *args, **kwargs)
                print(f"    ✓ {plot_name} completed successfully.")
                return result
            except Exception as e:
                print(f"    ❌ Error generating {plot_name}: {str(e)}")
                logging.error(f"Error in {plot_name}: {str(e)}", exc_info=True)
                return None
        return wrapper
    return decorator


class ClusterVisualizationService:
    """
    A service dedicated to generating visualizations from pre-computed clustering results.
    Optimized for performance and robust error handling.
    """
    
    # Class-level constants
    DEFAULT_PALETTE = [
        "#EF4444", "#22C55E", "#EAB308", "#3B82F6", "#8B5CF6",
        "#F97316", "#14B8A6", "#EC4899", "#6366F1", "#6B7280"
    ]
    
    def __init__(
        self,
        scaled_df: pd.DataFrame,
        preprocessed_df: pd.DataFrame,
        labels: np.ndarray,
        model: Any,
        silhouette_avg: float,
        algorithm_name: str,
        output_path: Path
    ):
        """
        Initializes the visualization service with clustering results.

        Args:
            scaled_df: DataFrame with scaled features used for clustering.
            preprocessed_df: Original preprocessed data for price analysis.
            labels: The cluster labels assigned to each data point.
            model: The fitted model from scikit-learn.
            silhouette_avg: The average silhouette score for the clustering.
            algorithm_name: The name of the algorithm used.
            output_path: The directory path to save all visualization files.
        """
        self.scaled_df = scaled_df
        self.X = scaled_df.drop(columns=["City"])
        self.preprocessed_df = preprocessed_df
        self.labels = labels
        self.model = model
        self.silhouette_avg = silhouette_avg
        self.algorithm_name = algorithm_name
        self.output_path = output_path
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Pre-compute commonly used data
        self._cluster_map = None
        self._merged_df = None
        self._cluster_colors = None
        self._unique_clusters = sorted(np.unique(self.labels).astype(int).tolist())
        
        print(f"--- Visualization Service Initialized for '{algorithm_name}' ---")
        print(f"Output will be saved to: {self.output_path}")
        print(f"Number of clusters: {len(self._unique_clusters)}")

    @property
    def cluster_map(self) -> pd.DataFrame:
        """Lazy-load cluster map."""
        if self._cluster_map is None:
            self._cluster_map = pd.DataFrame({
                "City": self.scaled_df["City"], 
                "Cluster": self.labels
            })
        return self._cluster_map

    @property
    def merged_df(self) -> pd.DataFrame:
        """Lazy-load merged dataframe."""
        if self._merged_df is None:
            self._merged_df = self.preprocessed_df.merge(
                self.cluster_map, on="City", how="left"
            )
        return self._merged_df

    @property
    def cluster_colors(self) -> Dict[int, str]:
        """Lazy-load cluster color mapping."""
        if self._cluster_colors is None:
            self._cluster_colors = self._get_cluster_color_mapping(self._unique_clusters)
        return self._cluster_colors

    def _get_cluster_color_mapping(self, clusters: List[int]) -> Dict[int, str]:
        """Map cluster ids to hex colors, cycling when clusters exceed palette length."""
        return {cid: self.DEFAULT_PALETTE[idx % len(self.DEFAULT_PALETTE)] 
                for idx, cid in enumerate(clusters)}

    def generate_all_visualizations(self, temp_dir: Path = None, parallel: bool = False) -> Dict[str, Path]:
        """
        Runs all visualization generation methods in sequence or parallel.
        Returns dictionary of chart types to file paths (only successful ones).
        
        Args:
            temp_dir: Directory to save temporary files.
            parallel: Whether to use parallel processing (experimental).
        """
        if temp_dir is None:
            temp_dir = self.output_path
            
        print("\n🚀 Starting visualization generation...")
        
        image_paths = {}
        
        # Define all visualization tasks
        viz_tasks = [
            ('silhouette', 'Silhouette Plot', self._generate_silhouette_visualization, 
             temp_dir / "silhouette_plot.png"),
            ('scatter', '2D Scatter Plot', self._generate_scatter_plot, 
             temp_dir / "scatter_plot.png"),
            ('boxplot', 'Box Plot', self._generate_box_plot, 
             temp_dir / "boxplot.png"),
            ('linechart', 'Line Chart', self._generate_line_chart, 
             temp_dir / "linechart.png"),
            ('radar', 'Radar Chart', self._generate_radar_chart, 
             temp_dir / "radar.png"),
        ]
        
        # Generate plots sequentially (more stable for matplotlib)
        for key, name, func, output_path in viz_tasks:
            result = func(output_path)
            if result is not None and output_path.exists():
                image_paths[key] = output_path
        
        # Generate map (special handling)
        map_path = temp_dir / "map.png"
        result = self._generate_map(map_path)
        if result is not None and map_path.exists():
            image_paths['map'] = map_path
        
        # Generate summary (doesn't produce image)
        self._generate_cluster_summary()
        
        print(f"\n✅ Successfully generated {len(image_paths)}/{len(viz_tasks) + 1} visualizations!")
        return image_paths

    @safe_plot_generation("Silhouette Plot")
    def _generate_silhouette_visualization(self, output_path: Path) -> Path:
        """Generate silhouette plot with error handling."""
        if self.algorithm_name in ["kmeans"] and hasattr(self.model, 'cluster_centers_'):
            visualizer = SilhouetteVisualizer(
                self.model, 
                colors=[self.cluster_colors[c] for c in self._unique_clusters]
            )
            visualizer.fit(self.X)
            visualizer.show(outpath=str(output_path))
            plt.close()
        else:
            # Generic fallback
            fig, ax = plt.subplots(1, 1, figsize=(10, 7))
            n_clusters = len(self._unique_clusters)
            ax.set_ylim([0, len(self.X) + (n_clusters + 1) * 10])
            
            sample_silhouette_values = silhouette_samples(self.X, self.labels)
            y_lower = 10
            
            for i in range(n_clusters):
                ith_cluster_silhouette_values = sample_silhouette_values[self.labels == i]
                ith_cluster_silhouette_values.sort()
                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i
                
                color = self.cluster_colors.get(i, '#6B7280')
                ax.fill_betweenx(
                    np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                    facecolor=color, edgecolor=color, alpha=0.7
                )
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
        
        return output_path

    @safe_plot_generation("2D Scatter Plot")
    def _generate_scatter_plot(self, output_path: Path) -> Path:
        """Generate 2D scatter plot with dimensionality reduction if needed."""
        data_to_plot = self.X.values if isinstance(self.X, pd.DataFrame) else self.X
        feature_names = self.X.columns.tolist() if isinstance(self.X, pd.DataFrame) else \
                        [f"Feature {i+1}" for i in range(data_to_plot.shape[1])]
        
        n_features = data_to_plot.shape[1]
        
        if n_features < 2:
            print(f" ⚠️ Skipping: Only {n_features} feature(s) available.")
            return None
        
        if n_features == 2:
            X_reduced = data_to_plot
            plot_method = "Original Features"
            xlabel, ylabel = feature_names[0], feature_names[1]
            print(" ℹ️ Plotting original 2 features directly.")
        else:
            pca = PCA(n_components=2, random_state=42)
            X_reduced = pca.fit_transform(data_to_plot)
            var1, var2 = pca.explained_variance_ratio_ * 100
            plot_method = "PCA"
            xlabel = f"PC1 ({var1:.1f}% var.)"
            ylabel = f"PC2 ({var2:.1f}% var.)"
            print(f" ℹ️ Reduced to 2D via PCA (total var: {var1+var2:.1f}%).")
        
        # Create plot
        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            x=X_reduced[:, 0], y=X_reduced[:, 1],
            hue=self.labels,
            palette=self.cluster_colors,
            s=80, alpha=0.9, edgecolor='k', legend='full'
        )
        
        plt.title(f"{plot_method} 2D Cluster Visualization", fontsize=16)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(title="Cluster")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path

    @safe_plot_generation("Box Plot")
    def _generate_box_plot(self, output_path: Path) -> Path:
        """Generate box plot for price distribution."""
        merged_df = self.merged_df.copy()
        
        if 'Year' not in merged_df.columns:
            merged_df['Year'] = pd.to_datetime(merged_df['Date']).dt.year
            
        # Ensure Year is numeric (not categorical) for min/max operations
        if pd.api.types.is_categorical_dtype(merged_df['Year']):
            merged_df['Year'] = merged_df['Year'].astype(int)
        elif not pd.api.types.is_numeric_dtype(merged_df['Year']):
            merged_df['Year'] = pd.to_numeric(merged_df['Year'], errors='coerce')

        commodities = sorted(merged_df["Commodity"].unique())
        n_commodities = len(commodities)
        
        print(f"📊 Plotting {n_commodities} commodities across {len(self._unique_clusters)} clusters")

        n_cols = 2
        n_rows = int(np.ceil(n_commodities / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 5))
        
        # Handle single subplot case
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        for i, commodity in enumerate(commodities):
            ax = axes[i]
            df_commodity = merged_df[merged_df['Commodity'] == commodity]
            
            if df_commodity.empty:
                ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center', fontsize=12)
                ax.set_title(commodity, fontsize=14, weight='bold')
                continue
            
            sns.boxplot(
                x='Year', y='Price', hue='Cluster',
                hue_order=self._unique_clusters,
                data=df_commodity, ax=ax,
                palette=self.cluster_colors,
                fliersize=3, linewidth=1.3, showmeans=True,
                meanprops=dict(marker='D', markerfacecolor='white', 
                             markeredgecolor='black', markersize=5)
            )
            
            ax.set_title(f"{commodity}", fontsize=14, weight='bold', pad=12)
            ax.set_xlabel("Year", fontsize=11, weight='semibold')
            ax.set_ylabel("Price (Rp)", fontsize=11, weight='semibold')
            ax.grid(True, axis='y', linestyle='--', alpha=0.3)
            
            if ax.get_legend():
                ax.get_legend().remove()
            
            if len(df_commodity['Year'].unique()) > 3:
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Remove empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        # Add legend
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles, labels, title='City Cluster',
                loc='upper center', bbox_to_anchor=(0.5, 0.99),
                ncol=min(len(self._unique_clusters), 6),
                fontsize=11, title_fontsize=12,
                frameon=True, fancybox=True, shadow=True, edgecolor='gray'
            )

        year_min, year_max = merged_df['Year'].min(), merged_df['Year'].max()
        fig.suptitle(
            f"Distribusi Harga per Komoditas ({year_min}–{year_max})", 
            fontsize=18, weight='bold', y=1.01
        )

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path

    @safe_plot_generation("Line Chart")
    def _generate_line_chart(self, output_path: Path) -> Path:
        """Generate monthly price trend line charts."""
        merged_df = self.merged_df.copy()
        
        # Prepare data
        merged_df["Date"] = pd.to_datetime(merged_df["Date"], errors="coerce")
        merged_df = merged_df.dropna(subset=["Date", "Price", "Cluster"]).copy()
        merged_df["Cluster"] = merged_df["Cluster"].astype(int)
        merged_df["YearMonth"] = merged_df["Date"].dt.to_period("M").dt.to_timestamp()

        commodities = sorted(merged_df["Commodity"].unique())[:10]
        
        cluster_agg_ym = (
            merged_df[merged_df["Commodity"].isin(commodities)]
            .groupby(["Commodity", "YearMonth", "Cluster"], as_index=False)["Price"]
            .mean()
        )

        # Create plot
        nrows, ncols = 5, 2
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 17))
        axes = axes.flatten()
        
        month_locator = mdates.MonthLocator(interval=6)
        year_locator = mdates.YearLocator()
        year_fmt = mdates.DateFormatter("%Y")

        for idx, commodity in enumerate(commodities):
            ax = axes[idx]
            data_c = cluster_agg_ym[cluster_agg_ym["Commodity"] == commodity]
            
            for cl in self._unique_clusters:
                sub = data_c[data_c["Cluster"] == cl].sort_values("YearMonth")
                if not sub.empty:
                    ax.plot(
                        sub["YearMonth"], sub["Price"],
                        label=f"Cluster {cl}",
                        color=self.cluster_colors[cl],
                        marker="o", linewidth=1.5, markersize=3
                    )
            
            ax.set_title(commodity, fontsize=12, weight='bold')
            ax.grid(True, which='both', linestyle='--', alpha=0.4)
            ax.xaxis.set_major_locator(year_locator)
            ax.xaxis.set_major_formatter(year_fmt)
            ax.xaxis.set_minor_locator(month_locator)
            
            if idx % ncols == 0:
                ax.set_ylabel("Average Price (Rp)")
            
            ax.legend(loc="best", fontsize=9)

        for k in range(len(commodities), nrows * ncols):
            fig.delaxes(axes[k])

        fig.suptitle("Monthly Average Price Trends by Cluster", y=1.0, fontsize=16, weight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path

    @safe_plot_generation("Radar Chart")
    def _generate_radar_chart(self, output_path: Path) -> Path:
        """Generate radar chart for commodity price profiles."""
        merged_df = self.merged_df
        
        # Check if we have enough commodities for a meaningful radar chart
        unique_commodities = merged_df['Commodity'].nunique()
        if unique_commodities <= 2:
            print(f"  ⚠️ Skipping radar chart: Only {unique_commodities} commodities (minimum 3 required for meaningful radar chart)")
            return None
        
        # Prepare data
        avg_prices = merged_df.groupby(['Cluster', 'Commodity'])['Price'].mean().reset_index()
        avg_prices['min_val'] = avg_prices.groupby('Commodity')['Price'].transform('min')
        avg_prices['max_val'] = avg_prices.groupby('Commodity')['Price'].transform('max')
        
        range_val = avg_prices['max_val'] - avg_prices['min_val']
        avg_prices['normalized'] = np.where(
            range_val > 0,
            (avg_prices['Price'] - avg_prices['min_val']) / range_val,
            0.5
        )
        
        radar_df = avg_prices.pivot(index='Cluster', columns='Commodity', values='normalized')
        
        # Create plot
        commodities = radar_df.columns.tolist()
        num_vars = len(commodities)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))

        for cluster_id, row in radar_df.iterrows():
            values = row.tolist()
            values += values[:1]
            color = self.cluster_colors[int(cluster_id)]
            ax.plot(angles, values, linewidth=2, label=f'Cluster {cluster_id}', color=color)
            ax.fill(angles, values, alpha=0.25, color=color)

        ax.set_yticks([0, 0.5, 1.0])
        ax.set_yticklabels(["Lowest Price", "Median Price", "Highest Price"])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(commodities, size=12)

        plt.title('Normalized Commodity Price Profile by Cluster', size=20, y=1.1)
        plt.legend(title='Clusters', loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path

    @safe_plot_generation("Map Visualization")
    def _generate_map(self, output_path: Path) -> Path:
        """Generate map visualization with fallback options."""
        coordinates_file = Path("data/city_coordinates.json")
        
        try:
            df_for_map = self._prepare_data_for_map(self.cluster_map, coordinates_file)
            if df_for_map.empty:
                print("  ⚠️ No map data available")
                return None
            
            # Try Folium first
            try:
                return self._create_cluster_map_png(df_for_map, output_path)
            except Exception as e:
                print(f"  ⚠️ Folium failed: {e}, trying static map...")
                return self._create_static_map(df_for_map, output_path)
                
        except Exception as e:
            print(f"  ❌ Map generation failed: {e}")
            return None

    def _prepare_data_for_map(self, cluster_data: pd.DataFrame, coordinates_filepath: Path) -> pd.DataFrame:
        """Consolidates city, cluster, and coordinate data."""
        try:
            with open(coordinates_filepath, 'r') as f:
                city_coordinates = json.load(f)
            
            df_coords = pd.DataFrame(city_coordinates.items(), columns=['City', 'Coordinates'])
            df_coords[['Lat', 'Lon']] = pd.DataFrame(df_coords['Coordinates'].tolist(), index=df_coords.index)
            df_coords.drop('Coordinates', axis=1, inplace=True)
            
            df_city_clusters = cluster_data[['City', 'Cluster']].drop_duplicates().reset_index(drop=True)
            df_map_data = pd.merge(df_city_clusters, df_coords, on='City', how='inner')
            
            print(f"  📍 Found coordinates for {len(df_map_data)} cities")
            return df_map_data
            
        except FileNotFoundError:
            print(f"  ❌ File not found: {coordinates_filepath}")
            return pd.DataFrame()
        except Exception as e:
            print(f"  ❌ Error preparing map data: {e}")
            return pd.DataFrame()

    def _create_cluster_map_png(self, df_map_data: pd.DataFrame, output_path: Path) -> Path:
        """Create cluster map and save as PNG using Selenium."""
        # Create folium map
        map_center = [-2.548926, 118.0148634]
        cluster_map = folium.Map(location=map_center, zoom_start=5, tiles="OpenStreetMap")

        # Add markers
        for _, row in df_map_data.iterrows():
            cluster_id = int(row['Cluster'])
            folium.CircleMarker(
                location=[row['Lat'], row['Lon']],
                radius=8,
                color=self.cluster_colors.get(cluster_id, '#808080'),
                fill=True,
                fill_color=self.cluster_colors.get(cluster_id, '#808080'),
                fill_opacity=0.7,
                weight=2,
                popup=folium.Popup(f"<b>{row['City']}</b><br>Cluster: {cluster_id}", max_width=200)
            ).add_to(cluster_map)

        # Add legend
        legend_html = '''
        <div style="position: fixed; bottom: 50px; right: 50px; width: 180px; 
                    background-color: white; z-index:9999; font-size:14px;
                    border:2px solid grey; border-radius: 5px; padding: 10px">
        <p style="margin-bottom: 5px; font-weight: bold;">City Clusters</p>
        '''
        for cluster_id in self._unique_clusters:
            color = self.cluster_colors[cluster_id]
            legend_html += f'''
            <p style="margin: 3px 0;">
                <span style="background-color:{color}; width: 20px; height: 20px; 
                            display: inline-block; border: 1px solid black; margin-right: 5px;"></span>
                Cluster {cluster_id}
            </p>
            '''
        legend_html += '</div>'
        cluster_map.get_root().html.add_child(folium.Element(legend_html))

        # Save as HTML first
        temp_html_path = output_path.parent / "temp_map.html"
        cluster_map.save(str(temp_html_path))
        
        try:
            # Convert to PNG
            options = Options()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-gpu')
            options.add_argument('--window-size=1200,800')
            
            driver = webdriver.Chrome(options=options)
            driver.set_page_load_timeout(30)
            driver.get(f'file://{temp_html_path.absolute()}')
            
            import time
            time.sleep(5)
            
            driver.save_screenshot(str(output_path))
            driver.quit()
            
            temp_html_path.unlink()
            return output_path
            
        except Exception as e:
            print(f"  ❌ Selenium error: {e}")
            if temp_html_path.exists():
                temp_html_path.unlink()
            raise

    def _create_static_map(self, df_map_data: pd.DataFrame, output_path: Path) -> Path:
        """Create static map using matplotlib (fallback)."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for _, row in df_map_data.iterrows():
            cluster_id = int(row['Cluster'])
            color = self.cluster_colors.get(cluster_id, '#6B7280')
            
            ax.scatter(row['Lon'], row['Lat'], c=[color], s=100, alpha=0.7, 
                      edgecolors='black', linewidth=1)
            ax.annotate(row['City'], (row['Lon'], row['Lat']), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.8)
        
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('City Clusters Geographic Distribution')
        ax.grid(True, alpha=0.3)
        
        for cluster_id in self._unique_clusters:
            color = self.cluster_colors[cluster_id]
            ax.scatter([], [], c=[color], s=100, label=f'Cluster {cluster_id}')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path

    def _generate_cluster_summary(self):
        """Generate cluster summary Excel file."""
        try:
            df_cluster_summary = self.cluster_map.groupby("Cluster")['City'].agg(
                number_of_members='count',
                members=lambda cities: ', '.join(sorted(cities))
            ).reset_index().sort_values("Cluster").set_index("Cluster")
            
            output_path = self.output_path / "cluster_members_summary.xlsx"
            df_cluster_summary.to_excel(output_path)
            print(f"  ✓ Cluster summary saved to {output_path.name}")
        except Exception as e:
            print(f"  ⚠️ Could not generate cluster summary: {e}")

    def generate_pdf_report(self, output_pdf_path: Path) -> Optional[Path]:
        """
        Generate PDF report with all available visualizations.
        Continues even if some plots failed to generate.
        """
        print("\n📄 Generating PDF report...")
        
        temp_plots_dir = output_pdf_path.parent / "temp_plots"
        temp_plots_dir.mkdir(exist_ok=True)
        
        try:
            # Generate all visualizations
            image_paths = self.generate_all_visualizations(temp_plots_dir)
            
            if not image_paths:
                print("❌ No visualizations were generated. Cannot create PDF.")
                return None
            
            # Create PDF
            doc = SimpleDocTemplate(str(output_pdf_path), pagesize=A4)
            story = []
            styles = getSampleStyleSheet()
            max_img_width = doc.width
            max_img_height = 4.5 * inch

            def make_image_flowable(img_path: Path) -> Optional[Image]:
                """Create an Image flowable with proper scaling."""
                try:
                    ir = ImageReader(str(img_path))
                    orig_w, orig_h = ir.getSize()
                    if orig_w == 0 or orig_h == 0:
                        return Image(str(img_path), width=max_img_width, height=max_img_height)
                    scale_w = max_img_width / orig_w
                    scale_h = max_img_height / orig_h
                    scale = min(scale_w, scale_h, 1.0)
                    new_w = orig_w * scale
                    new_h = orig_h * scale
                    return Image(str(img_path), width=new_w, height=new_h)
                except Exception as e:
                    print(f"  ⚠️ Could not process image {img_path.name}: {e}")
                    return None
            
            # Title
            title = Paragraph("Clustering Analysis Report", styles['Title'])
            story.append(title)
            story.append(Spacer(12, 12))
            
            # Add metadata
            metadata = f"""
            <b>Algorithm:</b> {self.algorithm_name.title()}<br/>
            <b>Number of Clusters:</b> {len(self._unique_clusters)}<br/>
            <b>Silhouette Score:</b> {self.silhouette_avg:.4f}<br/>
            <b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            story.append(Paragraph(metadata, styles['Normal']))
            story.append(Spacer(12, 24))
            
            # Define chart ordering
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
                    img = make_image_flowable(image_path)
                    if img is not None:
                        # Force scatter plot to start on new page
                        if title_text == "PCA Scatter Plot":
                            story.append(PageBreak())
                        
                        heading = Paragraph(title_text, styles['Heading2'])
                        story.append(KeepTogether([heading, Spacer(1, 6), img]))
                        story.append(Spacer(12, 12))
                        charts_added += 1
                else:
                    print(f"  ⚠️ Skipping {title_text} - not available")
            
            if charts_added == 0:
                story.append(Paragraph("No visualizations could be generated.", styles['Normal']))
                print("  ⚠️ No charts were added to the PDF")
            
            # Build PDF
            doc.build(story)
            
            # Clean up temporary plots
            import shutil
            if temp_plots_dir.exists():
                shutil.rmtree(temp_plots_dir)
            
            print(f"✅ PDF report saved as '{output_pdf_path.name}' with {charts_added} charts")
            return output_pdf_path
            
        except Exception as e:
            print(f"❌ Error generating PDF: {e}")
            logging.error(f"PDF generation error: {e}", exc_info=True)
            
            # Clean up on error
            if temp_plots_dir.exists():
                import shutil
                shutil.rmtree(temp_plots_dir)
            
            return None


def main():
    """
    Example workflow demonstrating the refactored visualization service.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # --- 1. Load Data ---
    SCALED_DATA_PATH = Path("data/master/features/master_food_prices_scaled.csv")
    PREPROCESSED_DATA_PATH = Path("data/master/features/master_food_prices.parquet")
    
    print("Loading data...")
    scaled_df = pd.read_csv(SCALED_DATA_PATH)
    preprocessed_df = pd.read_parquet(PREPROCESSED_DATA_PATH)
    
    # Clean data
    if "Unnamed: 0" in scaled_df.columns:
        scaled_df = scaled_df.drop(columns=["Unnamed: 0"])
    
    X = scaled_df.drop(columns=["City"])
    
    # --- 2. Run Clustering ---
    algorithm_to_run = "kmeans"
    n_clusters_to_run = 4

    print(f"\nRunning {algorithm_to_run.upper()} with K={n_clusters_to_run}...")
    
    model = KMeans(n_clusters=n_clusters_to_run, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    silhouette_avg = silhouette_score(X, labels)
    
    print(f"✓ Clustering complete. Silhouette Score: {silhouette_avg:.4f}")

    # --- 3. Generate Visualizations ---
    output_directory = SAVE_DIR / algorithm_to_run / f"k{n_clusters_to_run}"
    
    viz_service = ClusterVisualizationService(
        scaled_df=scaled_df,
        preprocessed_df=preprocessed_df,
        labels=labels,
        model=model,
        silhouette_avg=silhouette_avg,
        algorithm_name=algorithm_to_run,
        output_path=output_directory
    )
    
    # Generate all visualizations
    image_paths = viz_service.generate_all_visualizations()
    
    # Generate PDF report
    pdf_path = output_directory / f"clustering_report_{algorithm_to_run}_k{n_clusters_to_run}.pdf"
    viz_service.generate_pdf_report(pdf_path)
    
    print(f"\n{'='*60}")
    print(f"✅ All tasks completed!")
    print(f"📁 Output directory: {output_directory}")
    print(f"📊 Visualizations generated: {len(image_paths)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()