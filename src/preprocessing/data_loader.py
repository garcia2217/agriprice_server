"""
Data loading functionality for raw Excel files.
"""

import logging
import zipfile
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import pandas as pd

from .config import ConsolidationConfig

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Handles discovery and loading of raw Excel files from directories or ZIP files.
    """
    
    def __init__(self, config: ConsolidationConfig):
        """
        Initialize DataLoader with configuration.
        
        Args:
            config: Configuration object with data selection parameters
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._temp_dir = None  # For ZIP extraction
    
    def __del__(self):
        """Clean up temporary directory if it exists."""
        self.cleanup_temp_dir()
    
    def cleanup_temp_dir(self):
        """Clean up temporary directory used for ZIP extraction."""
        if self._temp_dir and Path(self._temp_dir).exists():
            shutil.rmtree(self._temp_dir)
            self._temp_dir = None
    
    def extract_zip_file(self, zip_path: Path) -> Path:
        """
        Extract ZIP file to temporary directory and find the actual data root.
        
        Args:
            zip_path: Path to ZIP file
            
        Returns:
            Path to the actual data directory (handles nested structures)
            
        Raises:
            ValueError: If ZIP file is invalid or extraction fails
        """
        if not zip_path.exists():
            raise FileNotFoundError(f"ZIP file not found: {zip_path}")
        
        if not zipfile.is_zipfile(zip_path):
            raise ValueError(f"Invalid ZIP file: {zip_path}")
        
        # Clean up any existing temp directory
        self.cleanup_temp_dir()
        
        # Create new temporary directory
        self._temp_dir = tempfile.mkdtemp(prefix="food_price_data_")
        temp_path = Path(self._temp_dir)
        
        self.logger.info(f"Extracting ZIP file to: {temp_path}")
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_path)
            
            self.logger.info(f"Successfully extracted ZIP file: {zip_path}")
            
            # Find the actual data root directory (handles nested ZIP structures)
            data_root = self._find_data_root(temp_path)
            self.logger.info(f"Data root directory: {data_root}")
            
            return data_root
            
        except Exception as e:
            self.cleanup_temp_dir()
            raise ValueError(f"Failed to extract ZIP file {zip_path}: {str(e)}")
    
    def _find_data_root(self, extracted_path: Path) -> Path:
        """
        Find the actual data root directory in extracted ZIP.
        Handles cases where ZIP has an extra wrapper directory.
        
        Args:
            extracted_path: Path where ZIP was extracted
            
        Returns:
            Path to the directory containing Province/City structure
        """
        # Check if current path has province structure
        if self._has_province_structure(extracted_path):
            return extracted_path
        
        # Look for subdirectories that might contain the data
        subdirs = [item for item in extracted_path.iterdir() if item.is_dir()]
        
        for subdir in subdirs:
            if self._has_province_structure(subdir):
                self.logger.info(f"Found data root in subdirectory: {subdir.name}")
                return subdir
        
        # If no valid structure found, return original path and let validation catch it later
        self.logger.warning(f"No valid Province/City structure found in ZIP. Using: {extracted_path}")
        return extracted_path
    
    def _has_province_structure(self, path: Path) -> bool:
        """
        Check if a path contains the expected Province/City/Excel structure.
        
        Args:
            path: Path to check
            
        Returns:
            True if path contains valid province structure
        """
        if not path.exists() or not path.is_dir():
            return False
        
        # Look for at least one directory that could be a province
        province_dirs = [item for item in path.iterdir() if item.is_dir()]
        
        if not province_dirs:
            return False
        
        # Check if at least one province directory has the expected city/excel structure
        for province_dir in province_dirs:
            city_dirs = [item for item in province_dir.iterdir() if item.is_dir()]
            
            for city_dir in city_dirs:
                excel_files = list(city_dir.glob("*.xlsx"))
                if excel_files:
                    # Found at least one Excel file in a City directory
                    return True
        
        return False
    
    def get_data_path(self, input_path: Union[str, Path]) -> Path:
        """
        Get the data path, extracting ZIP if necessary.
        
        Args:
            input_path: Path to directory or ZIP file
            
        Returns:
            Path to data directory (extracted if ZIP)
        """
        input_path = Path(input_path)
        
        if self.config.input_type == "zip" or (input_path.suffix.lower() == '.zip'):
            return self.extract_zip_file(input_path)
        else:
            return input_path
    
    def discover_available_data(self, raw_data_path: Path) -> Dict[str, List[str]]:
        """
        Discover all available provinces and years in the raw data directory.
        
        Args:
            raw_data_path: Path to raw data directory
            
        Returns:
            Dictionary with 'provinces' and 'years' lists
        """
        available_provinces = []
        available_years = set()
        
        if raw_data_path.exists():
            for province_path in raw_data_path.iterdir():
                if province_path.is_dir():
                    available_provinces.append(province_path.name)
                    
                    # Check for years in this province by examining Excel files
                    for city_path in province_path.iterdir():
                        if city_path.is_dir():
                            for file_path in city_path.glob("*.xlsx"):
                                try:
                                    # Extract years from date columns instead of filename
                                    years_from_file = self._extract_years_from_file(file_path)
                                    available_years.update(years_from_file)
                                except Exception as e:
                                    self.logger.warning(f"Could not extract years from {file_path}: {e}")
                                    continue
        
        return {
            'provinces': sorted(available_provinces),
            'years': sorted(list(available_years))
        }
    
    def _extract_years_from_file(self, file_path: Path) -> List[str]:
        """
        Extract years from date columns in an Excel file.
        
        Args:
            file_path: Path to Excel file
            
        Returns:
            List of years found in the file
        """
        try:
            # Read only the first row to get column names (dates)
            df = pd.read_excel(file_path, nrows=0)
            
            years = set()
            for col in df.columns:
                if isinstance(col, str) and '/' in col:
                    try:
                        # Parse date column (format: "DD/ MM/ YYYY")
                        date_obj = pd.to_datetime(col, format=self.config.date_format, errors='coerce')
                        if pd.notna(date_obj):
                            years.add(str(date_obj.year))
                    except:
                        continue
            
            return sorted(list(years))
            
        except Exception as e:
            self.logger.debug(f"Error extracting years from {file_path}: {e}")
            return []
    
    def discover_data_files(self, raw_data_path: Path) -> List[Dict[str, Any]]:
        """
        Discover all Excel files matching the configuration criteria.
        
        Args:
            raw_data_path: Path to raw data directory
            
        Returns:
            List of dictionaries containing file metadata
            
        Raises:
            FileNotFoundError: If raw data path does not exist
        """
        discovered_files = []
        
        if not raw_data_path.exists():
            raise FileNotFoundError(f"Raw data path does not exist: {raw_data_path}")
        
        for province_path in raw_data_path.iterdir():
            if not province_path.is_dir():
                continue
                
            if province_path.name not in self.config.provinces:
                continue
                
            self.logger.info(f"Processing province: {province_path.name}")
            
            for city_path in province_path.iterdir():
                if not city_path.is_dir():
                    continue
                    
                city_files = []
                for file_path in city_path.glob("*.xlsx"):
                    try:
                        # Extract years from the file content instead of filename
                        years_in_file = self._extract_years_from_file(file_path)
                        
                        # Check if any of the years in the file match our configuration
                        matching_years = [year for year in years_in_file if year in self.config.years]
                        
                        if matching_years:
                            # Use the first matching year or all years if multiple
                            primary_year = matching_years[0]
                            
                            file_info = {
                                'city': city_path.name,
                                'year': primary_year,
                                'years_in_file': years_in_file,  # Store all years found
                                'file_path': file_path,
                                'file_size': file_path.stat().st_size,
                                'province': province_path.name,
                                'filename': file_path.name
                            }
                            city_files.append(file_info)
                            discovered_files.append(file_info)
                    
                    except Exception as e:
                        self.logger.warning(f"Could not process file {file_path}: {e}")
                        continue
                
                if city_files:
                    self.logger.info(f"  Found {len(city_files)} files for city: {city_path.name}")
        
        self.logger.info(f"Total files discovered: {len(discovered_files)}")
        return discovered_files
    
    def load_excel_file(self, file_info: Dict[str, Any]) -> pd.DataFrame:
        """
        Load and perform initial processing of a single Excel file.
        
        Args:
            file_info: Dictionary containing file metadata
            
        Returns:
            Raw DataFrame from Excel file
            
        Raises:
            ValueError: If file cannot be processed
        """
        try:
            # Load Excel file
            df = pd.read_excel(file_info['file_path'])
            self.logger.debug(f"Loaded {file_info['file_path']}: {df.shape}")
            
            # Validate required columns exist
            if self.config.commodity_column_name not in df.columns:
                raise ValueError(
                    f"Required column '{self.config.commodity_column_name}' not found in {file_info['file_path']}"
                )
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading file {file_info['file_path']}: {str(e)}")
            raise ValueError(f"Failed to load {file_info['file_path']}: {str(e)}")
    
    def transform_to_long_format(self, df: pd.DataFrame, file_info: Dict[str, Any]) -> pd.DataFrame:
        """
        Transform DataFrame from wide to long format and add metadata.
        
        Args:
            df: Wide format DataFrame from Excel
            file_info: File metadata dictionary
            
        Returns:
            Long format DataFrame with metadata columns
        """
        # Drop specified columns
        columns_to_drop = [col for col in self.config.columns_to_drop if col in df.columns]
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
        
        # Rename commodity column
        df = df.rename(columns={self.config.commodity_column_name: "Commodity"})
        
        # Add metadata columns (Year will be extracted from date columns later)
        df["City"] = file_info['city']
        df["Province"] = file_info['province']
        df["Source_File"] = file_info.get('filename', 'unknown')
        
        # Transform to long format
        id_vars = ["Commodity", "City", "Province", "Source_File"]
        long_df = df.melt(
            id_vars=id_vars,
            var_name="Date",
            value_name="Price"
        )
        
        # Extract year and month from date column after melting
        long_df = self._extract_date_info(long_df)
        
        return long_df
    
    def _extract_date_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract year, month, and proper date from the Date column.
        
        Args:
            df: DataFrame with Date column containing date strings
            
        Returns:
            DataFrame with additional Year, Month columns and parsed Date
        """
        df_copy = df.copy()
        
        # Parse dates and extract year/month
        date_parsed = pd.to_datetime(df_copy['Date'], format=self.config.date_format, errors='coerce')
        
        df_copy['Year'] = date_parsed.dt.year.astype(str)
        df_copy['Month'] = date_parsed.dt.month
        df_copy['Date'] = date_parsed
        
        # Remove rows with invalid dates
        df_copy = df_copy.dropna(subset=['Date'])
        
        return df_copy
    
    def load_and_transform_file(self, file_info: Dict[str, Any]) -> pd.DataFrame:
        """
        Load Excel file and transform to long format in one step.
        
        Args:
            file_info: Dictionary containing file metadata
            
        Returns:
            Cleaned DataFrame in long format
        """
        df = self.load_excel_file(file_info)
        return self.transform_to_long_format(df, file_info)
