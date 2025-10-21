"""
File processing service for handling file I/O operations.

This service manages file processing, temporary file cleanup,
and data persistence with proper resource management.
"""

import time
import shutil
from pathlib import Path
from typing import Optional
import pandas as pd

from ..constants import MAX_VALIDATION_AGE_HOURS


class FileProcessingService:
    """Service for file processing and data persistence."""
    
    def __init__(self):
        """Initialize file processing service."""
        self.temp_data_dir = Path("temp_data")
        self.temp_data_dir.mkdir(exist_ok=True)
    
    def process_zip_file(self, file, validation_id: str) -> None:
        """
        Process ZIP file and prepare for validation.
        
        Args:
            file: Uploaded file object
            validation_id: Unique validation identifier
        """
        # Reset file position for processing
        file.seek(0)
        
        # The actual ZIP processing is handled by DataConsolidator
        # This method is a placeholder for any additional file processing
        # that might be needed in the future
        pass
    
    def save_validated_data(self, df: pd.DataFrame, validation_id: str) -> Path:
        """
        Save validated data as Parquet file.
        
        Args:
            df: Validated dataframe
            validation_id: Unique validation identifier
            
        Returns:
            Path to saved Parquet file
        """
        parquet_path = self.temp_data_dir / f"validation_{validation_id}.parquet"
        
        try:
            df.to_parquet(parquet_path, index=False)
            return parquet_path
        except Exception as e:
            raise RuntimeError(f"Failed to save validated data: {str(e)}")
    
    def load_validated_data(self, validation_id: str) -> pd.DataFrame:
        """
        Load validated data from Parquet file.
        
        Args:
            validation_id: Unique validation identifier
            
        Returns:
            Loaded dataframe
            
        Raises:
            FileNotFoundError: If validation data not found
        """
        parquet_path = self.temp_data_dir / f"validation_{validation_id}.parquet"
        
        if not parquet_path.exists():
            raise FileNotFoundError(f"Validation data not found: {validation_id}")
        
        try:
            return pd.read_parquet(parquet_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load validated data: {str(e)}")
    
    def cleanup_old_files(self, hours: int = MAX_VALIDATION_AGE_HOURS) -> int:
        """
        Clean up validation files older than specified hours.
        
        Args:
            hours: Age threshold in hours
            
        Returns:
            Number of files cleaned up
        """
        if not self.temp_data_dir.exists():
            return 0
        
        cutoff_time = time.time() - (hours * 60 * 60)
        cleaned_count = 0
        
        try:
            for parquet_file in self.temp_data_dir.glob("validation_*.parquet"):
                if parquet_file.stat().st_mtime < cutoff_time:
                    parquet_file.unlink()
                    cleaned_count += 1
            
            if cleaned_count > 0:
                print(f"🧹 Cleaned up {cleaned_count} old validation files")
                
        except Exception as e:
            print(f"❌ Error during validation cleanup: {e}")
        
        return cleaned_count
    
    def get_validation_file_path(self, validation_id: str) -> Path:
        """
        Get the file path for a validation ID.
        
        Args:
            validation_id: Unique validation identifier
            
        Returns:
            Path to validation file
        """
        return self.temp_data_dir / f"validation_{validation_id}.parquet"
    
    def validation_file_exists(self, validation_id: str) -> bool:
        """
        Check if validation file exists.
        
        Args:
            validation_id: Unique validation identifier
            
        Returns:
            True if file exists, False otherwise
        """
        return self.get_validation_file_path(validation_id).exists()
    
    def delete_validation_file(self, validation_id: str) -> bool:
        """
        Delete a specific validation file.
        
        Args:
            validation_id: Unique validation identifier
            
        Returns:
            True if file was deleted, False if file didn't exist
        """
        file_path = self.get_validation_file_path(validation_id)
        
        if file_path.exists():
            try:
                file_path.unlink()
                return True
            except Exception as e:
                print(f"❌ Error deleting validation file {validation_id}: {e}")
                return False
        
        return False
