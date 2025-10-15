"""
Data consolidation functionality for combining multiple data sources.
"""

import logging
import json
import tempfile
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Union
import pandas as pd

from .config import ConsolidationConfig
from .data_loader import DataLoader
from .cleaner import DataCleaner

# Try to import validators, but handle the case where it's not available
try:
    from ..utils.validators import validate_processed_data
except ImportError:
    # Fallback validation function
    def validate_processed_data(df):
        return {
            'total_rows': len(df),
            'total_cities': df['City'].nunique() if 'City' in df.columns else 0,
            'total_commodities': df['Commodity'].nunique() if 'Commodity' in df.columns else 0,
            'quality_issues': []
        }

logger = logging.getLogger(__name__)


class DataConsolidator:
    """
    Orchestrates the complete data consolidation pipeline.
    """
    
    def __init__(self, config: ConsolidationConfig):
        """
        Initialize DataConsolidator with configuration.
        
        Args:
            config: Configuration object for consolidation pipeline
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize components
        self.data_loader = DataLoader(config)
        self.data_cleaner = DataCleaner(config)
    
    def setup_logging(self) -> logging.Logger:
        """
        Setup logging with optional file output.
        
        Returns:
            Configured logger instance
        """
        # Create logger
        logger = logging.getLogger("food_price_clustering")
        logger.setLevel(getattr(logging, self.config.log_level))
        
        # Clear any existing handlers
        logger.handlers.clear()
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Console handler (always enabled)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, self.config.log_level))
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)
        
        # File handler (optional)
        if self.config.enable_file_logging:
            # Create logs directory
            logs_dir = Path("logs")
            logs_dir.mkdir(exist_ok=True)
            
            # Generate timestamped log filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = logs_dir / f"data_consolidation_{timestamp}.log"
            
            file_handler = logging.FileHandler(log_filename, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)  # Log everything to file
            file_handler.setFormatter(detailed_formatter)
            logger.addHandler(file_handler)
            
            logger.info(f"📝 File logging enabled - Log file: {log_filename}")
        else:
            logger.info("📺 Console logging only (file logging disabled)")
        
        return logger
    
    def auto_populate_config(self, raw_data_path: Path) -> None:
        """
        Auto-populate empty configuration lists with available data.
        
        Args:
            raw_data_path: Path to raw data directory
        """
        available_data = self.data_loader.discover_available_data(raw_data_path)
        
        # Auto-populate empty lists with all available data
        if not self.config.provinces:
            self.config.provinces = available_data['provinces']
            self.logger.info(f"Auto-populated provinces: {len(self.config.provinces)} found")
            
        if not self.config.years:
            self.config.years = available_data['years']
            self.logger.info(f"Auto-populated years: {self.config.years}")
    
    def consolidate_data(self, input_path: Union[str, Path], remove_outliers: bool = False) -> pd.DataFrame:
        """
        Main function to consolidate all food price data.
        
        Args:
            input_path: Path to raw data directory or ZIP file
            remove_outliers: Whether to remove price outliers during cleaning
            
        Returns:
            Consolidated DataFrame
            
        Raises:
            ValueError: If no files are found or processed successfully
        """
        self.logger.info("Starting data consolidation process")
        
        # Get the actual data path (extract ZIP if necessary)
        raw_data_path = self.data_loader.get_data_path(input_path)
        
        # Auto-populate configuration if needed
        self.auto_populate_config(raw_data_path)
        
        # Discover files
        discovered_files = self.data_loader.discover_data_files(raw_data_path)
        
        if not discovered_files:
            raise ValueError("No files found matching the configuration criteria")
        
        # Process each file
        all_dataframes = []
        failed_files = []
        
        for i, file_info in enumerate(discovered_files, 1):
            self.logger.info(f"Processing file {i}/{len(discovered_files)}: {file_info['city']} - {file_info['year']}")
            
            try:
                # Load and transform file
                df = self.data_loader.load_and_transform_file(file_info)
                
                # Clean the data
                df_clean = self.data_cleaner.clean_dataframe(df, remove_outliers=remove_outliers)
                
                all_dataframes.append(df_clean)
                
            except Exception as e:
                self.logger.error(f"Failed to process {file_info['file_path']}: {str(e)}")
                failed_files.append({
                    'file_path': str(file_info['file_path']),
                    'error': str(e)
                })
                continue
        
        if not all_dataframes:
            raise ValueError("No files were successfully processed")
        
        # Report failed files
        if failed_files:
            self.logger.warning(f"Failed to process {len(failed_files)} files:")
            for failed in failed_files:
                self.logger.warning(f"  - {failed['file_path']}: {failed['error']}")
        
        # Concatenate all dataframes
        self.logger.info("Concatenating all processed dataframes")
        consolidated_df = pd.concat(all_dataframes, ignore_index=True)
        
        # Final validation
        validation_results = validate_processed_data(consolidated_df)
        
        self.logger.info("Data consolidation completed successfully")
        self.logger.info(f"Final dataset: {validation_results['total_rows']} rows, "
                        f"{validation_results['total_cities']} cities, "
                        f"{validation_results['total_commodities']} commodities")
        
        # Log any quality issues
        if validation_results['quality_issues']:
            self.logger.warning("Quality issues detected:")
            for issue in validation_results['quality_issues']:
                self.logger.warning(f"  - {issue}")
        
        return consolidated_df
    
    def process_zip_upload(self, zip_file_bytes: bytes, remove_outliers: bool = False, 
                          cleanup_after: bool = True) -> Dict[str, Any]:
        """
        Process ZIP file from uploaded bytes (API-friendly).
        
        Args:
            zip_file_bytes: ZIP file content as bytes
            remove_outliers: Whether to remove price outliers during cleaning
            cleanup_after: Whether to cleanup temp files after processing
            
        Returns:
            Dictionary with processing results similar to run_full_pipeline
            
        Raises:
            ValueError: If ZIP processing fails
        """
        temp_zip_path = None
        
        try:
            # Create temporary file from bytes
            with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_zip:
                temp_zip.write(zip_file_bytes)
                temp_zip_path = Path(temp_zip.name)
            
            self.logger.info(f"Processing ZIP upload ({len(zip_file_bytes)} bytes)")
            
            # Process the temporary ZIP file
            consolidated_df = self.consolidate_data(temp_zip_path, remove_outliers)
            
            # Validate results
            validation_results = validate_processed_data(consolidated_df)
            
            return {
                'success': True,
                'consolidated_df': consolidated_df,
                'validation_results': validation_results,
                'config': self.config.model_dump(),
                'source_type': 'zip_upload'
            }
            
        except Exception as e:
            self.logger.error(f"ZIP upload processing failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'config': self.config.model_dump(),
                'source_type': 'zip_upload'
            }
            
        finally:
            # Cleanup temporary ZIP file
            if temp_zip_path and temp_zip_path.exists():
                temp_zip_path.unlink(missing_ok=True)
            
            # Cleanup extracted files if requested
            if cleanup_after:
                self.data_loader.cleanup_temp_dir()
    
    def process_zip_stream(self, zip_stream, remove_outliers: bool = False, 
                          cleanup_after: bool = True) -> Dict[str, Any]:
        """
        Process ZIP file from a file-like object (Flask/FastAPI upload).
        
        Args:
            zip_stream: File-like object containing ZIP data
            remove_outliers: Whether to remove price outliers during cleaning
            cleanup_after: Whether to cleanup temp files after processing
            
        Returns:
            Dictionary with processing results
        """
        try:
            # Read bytes from stream
            zip_stream.seek(0)  # Ensure we're at the beginning
            zip_bytes = zip_stream.read()
            
            self.logger.info(f"Processing ZIP stream ({len(zip_bytes)} bytes)")
            
            return self.process_zip_upload(zip_bytes, remove_outliers, cleanup_after)
            
        except Exception as e:
            self.logger.error(f"ZIP stream processing failed: {str(e)}")
            return {
                'success': False,
                'error': f"Failed to read ZIP stream: {str(e)}",
                'config': self.config.model_dump(),
                'source_type': 'zip_stream'
            }
    
    def process_zip_with_export(self, zip_file_bytes: bytes, output_dir: Path = None,
                               remove_outliers: bool = False) -> Dict[str, Any]:
        """
        Process ZIP upload and export results (complete API workflow).
        
        Args:
            zip_file_bytes: ZIP file content as bytes
            output_dir: Directory to save outputs
            remove_outliers: Whether to remove price outliers
            
        Returns:
            Dictionary with processing results and export paths
        """
        try:
            # Process ZIP without cleanup (we need the data for export)
            result = self.process_zip_upload(zip_file_bytes, remove_outliers, cleanup_after=False)
            
            if not result['success']:
                return result
            
            # Export the results
            export_paths = self.export_consolidated_data(result['consolidated_df'], output_dir)
            result['export_paths'] = export_paths
            
            return result
            
        except Exception as e:
            self.logger.error(f"ZIP processing with export failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'config': self.config.model_dump(),
                'source_type': 'zip_upload_with_export'
            }
            
        finally:
            # Always cleanup after export
            self.data_loader.cleanup_temp_dir()
    
    def export_consolidated_data(self, df: pd.DataFrame, output_dir: Path = None) -> Dict[str, str]:
        """
        Export consolidated DataFrame to multiple formats.
        
        Args:
            df: Consolidated DataFrame to export
            output_dir: Directory to save outputs (default: data/processed)
            
        Returns:
            Dictionary with export file paths
        """
        if output_dir is None:
            output_dir = Path("data/processed")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"food_prices_consolidated_{timestamp}"
        
        export_paths = {}
        
        # Export to CSV (most common format)
        csv_path = output_dir / f"{base_filename}.csv"
        df.to_csv(csv_path, index=False)
        export_paths['csv'] = str(csv_path)
        
        # Export to Excel (for easy viewing and sharing)
        excel_path = output_dir / f"{base_filename}.xlsx"
        df.to_excel(excel_path, index=False)
        export_paths['excel'] = str(excel_path)
        
        # Export metadata
        metadata = {
            'export_timestamp': datetime.now().isoformat(),
            'configuration': self.config.model_dump(),
            'data_summary': {
                'total_rows': len(df),
                'total_cities': df['City'].nunique(),
                'total_commodities': df['Commodity'].nunique(),
                'date_range': {
                    'min_date': df['Date'].min().isoformat(),
                    'max_date': df['Date'].max().isoformat()
                },
                'cities': sorted(df['City'].unique().tolist()),
                'commodities': sorted(df['Commodity'].unique().tolist())
            }
        }
        
        metadata_path = output_dir / f"{base_filename}_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, default=str)
        export_paths['metadata'] = str(metadata_path)
        
        self.logger.info(f"Data exported successfully:")
        for format_type, file_path in export_paths.items():
            file_size = Path(file_path).stat().st_size / 1024**2  # MB
            self.logger.info(f"  {format_type.upper()}: {file_path} ({file_size:.2f} MB)")
        
        return export_paths
    
    def run_full_pipeline(self, input_path: Union[str, Path], output_dir: Path = None, 
                         remove_outliers: bool = False) -> Dict[str, Any]:
        """
        Run the complete consolidation pipeline from raw data to exported files.
        
        Args:
            input_path: Path to raw data directory or ZIP file
            output_dir: Directory to save outputs
            remove_outliers: Whether to remove price outliers
            
        Returns:
            Dictionary with pipeline results and export paths
        """
        # Setup logging
        self.setup_logging()
        
        try:
            # Consolidate data
            consolidated_df = self.consolidate_data(input_path, remove_outliers)
            
            # Export data
            export_paths = self.export_consolidated_data(consolidated_df, output_dir)
            
            # Validate final results
            validation_results = validate_processed_data(consolidated_df)
            
            return {
                'success': True,
                'consolidated_df': consolidated_df,
                'export_paths': export_paths,
                'validation_results': validation_results,
                'config': self.config.model_dump()
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'config': self.config.model_dump()
            }
