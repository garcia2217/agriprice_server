"""
Parallel processing functionality for data consolidation pipeline.

This module provides parallel file processing capabilities using ThreadPoolExecutor
to significantly improve performance when processing multiple Excel files.
"""

import logging
import concurrent.futures
from multiprocessing import cpu_count
from typing import List, Dict, Any, Tuple
import pandas as pd

logger = logging.getLogger(__name__)


class ParallelFileProcessor:
    """
    Handles parallel processing of multiple Excel files for data consolidation.
    
    Uses ThreadPoolExecutor for I/O-bound operations like Excel file reading,
    which is optimal for this use case.
    """
    
    def __init__(self, max_workers: int = None):
        """
        Initialize parallel file processor.
        
        Args:
            max_workers: Maximum number of worker threads. 
                        Defaults to min(8, cpu_count()) for optimal performance.
        """
        self.max_workers = max_workers or min(8, cpu_count())
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        self.logger.info(f"Initialized ParallelFileProcessor with {self.max_workers} workers")
    
    def process_files_parallel(self, 
                             discovered_files: List[Dict[str, Any]], 
                             data_loader, 
                             data_cleaner,
                             max_workers: int = None) -> Tuple[List[pd.DataFrame], List[Dict[str, str]]]:
        """
        Process multiple files in parallel using ThreadPoolExecutor.
        
        Args:
            discovered_files: List of file metadata dictionaries
            data_loader: DataLoader instance for file loading
            data_cleaner: DataCleaner instance for data cleaning
            max_workers: Override max workers for this operation
            
        Returns:
            Tuple of (successful_dataframes, failed_files_info)
        """
        if not discovered_files:
            return [], []
        
        # Use provided max_workers or instance default
        workers = max_workers or self.max_workers
        workers = min(workers, len(discovered_files))  # Don't exceed number of files
        
        self.logger.info(f"Processing {len(discovered_files)} files with {workers} workers")
        
        def process_single_file(file_info: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, str]]:
            """
            Process a single file and return result or error info.
            
            Returns:
                Tuple of (dataframe, error_info) where one is None
            """
            try:
                # Load and transform file
                df = data_loader.load_and_transform_file(file_info)
                
                # Clean the data
                df_clean = data_cleaner.clean_dataframe(df, remove_outliers=False)
                
                self.logger.debug(f"Successfully processed: {file_info.get('filename', 'unknown')}")
                return (df_clean, None)
                
            except Exception as e:
                error_info = {
                    'file_path': str(file_info.get('file_path', 'unknown')),
                    'filename': file_info.get('filename', 'unknown'),
                    'city': file_info.get('city', 'unknown'),
                    'error': str(e)
                }
                self.logger.error(f"Failed to process {file_info.get('filename', 'unknown')}: {str(e)}")
                return (None, error_info)
        
        # Process files in parallel
        successful_dataframes = []
        failed_files = []
        
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                # Submit all tasks
                future_to_file = {
                    executor.submit(process_single_file, file_info): file_info 
                    for file_info in discovered_files
                }
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_file):
                    file_info = future_to_file[future]
                    
                    try:
                        result_df, error_info = future.result()
                        
                        if result_df is not None:
                            successful_dataframes.append(result_df)
                        else:
                            failed_files.append(error_info)
                            
                    except Exception as e:
                        # Handle unexpected errors in future execution
                        error_info = {
                            'file_path': str(file_info.get('file_path', 'unknown')),
                            'filename': file_info.get('filename', 'unknown'),
                            'city': file_info.get('city', 'unknown'),
                            'error': f"Unexpected error: {str(e)}"
                        }
                        failed_files.append(error_info)
                        self.logger.error(f"Unexpected error processing {file_info.get('filename', 'unknown')}: {str(e)}")
        
        except Exception as e:
            self.logger.error(f"Parallel processing failed: {str(e)}")
            # Fallback to sequential processing
            self.logger.warning("Falling back to sequential processing")
            return self._process_files_sequential(discovered_files, data_loader, data_cleaner)
        
        # Log results
        self.logger.info(f"Parallel processing completed: {len(successful_dataframes)} successful, {len(failed_files)} failed")
        
        if failed_files:
            self.logger.warning(f"Failed files summary:")
            for failed in failed_files[:5]:  # Show first 5 failures
                self.logger.warning(f"  - {failed['filename']}: {failed['error']}")
            if len(failed_files) > 5:
                self.logger.warning(f"  ... and {len(failed_files) - 5} more failures")
        
        return successful_dataframes, failed_files
    
    def _process_files_sequential(self, 
                                discovered_files: List[Dict[str, Any]], 
                                data_loader, 
                                data_cleaner) -> Tuple[List[pd.DataFrame], List[Dict[str, str]]]:
        """
        Fallback sequential processing when parallel processing fails.
        
        Args:
            discovered_files: List of file metadata dictionaries
            data_loader: DataLoader instance for file loading
            data_cleaner: DataCleaner instance for data cleaning
            
        Returns:
            Tuple of (successful_dataframes, failed_files_info)
        """
        self.logger.info("Processing files sequentially (fallback mode)")
        
        successful_dataframes = []
        failed_files = []
        
        for i, file_info in enumerate(discovered_files, 1):
            self.logger.info(f"Processing file {i}/{len(discovered_files)}: {file_info.get('filename', 'unknown')}")
            
            try:
                # Load and transform file
                df = data_loader.load_and_transform_file(file_info)
                
                # Clean the data
                df_clean = data_cleaner.clean_dataframe(df, remove_outliers=False)
                
                successful_dataframes.append(df_clean)
                
            except Exception as e:
                error_info = {
                    'file_path': str(file_info.get('file_path', 'unknown')),
                    'filename': file_info.get('filename', 'unknown'),
                    'city': file_info.get('city', 'unknown'),
                    'error': str(e)
                }
                failed_files.append(error_info)
                self.logger.error(f"Failed to process {file_info.get('filename', 'unknown')}: {str(e)}")
                continue
        
        self.logger.info(f"Sequential processing completed: {len(successful_dataframes)} successful, {len(failed_files)} failed")
        return successful_dataframes, failed_files
    
    def get_optimal_workers(self, num_files: int) -> int:
        """
        Calculate optimal number of workers for given number of files.
        
        Args:
            num_files: Number of files to process
            
        Returns:
            Optimal number of workers
        """
        # Don't use more workers than files
        optimal = min(self.max_workers, num_files)
        
        # Don't use more workers than CPU cores for I/O bound tasks
        optimal = min(optimal, cpu_count())
        
        return max(1, optimal)  # At least 1 worker
