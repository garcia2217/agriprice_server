from django.shortcuts import render
from django.http import HttpResponse
from pathlib import Path

def home(request):
    return HttpResponse("Halo dunia, ini Django pertama lo 🔥")

def dashboard(request):
    return HttpResponse("Halo dunia, gw dashboard")

def about(request):
    return HttpResponse("Halo duania, gw about")

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from src.preprocessing import ConsolidationConfig, DataConsolidator
from src.features import FeatureEngineeringConfig, FeatureEngineeringPipeline
from src.clustering import ClusteringPipelineConfig, ClusteringAnalysisPipeline
from src.utils.validators import (
    validate_file_format, validate_geographic_scope, validate_data_structure,
    validate_data_quality, validate_commodities, validate_temporal_range,
    extract_available_options, calculate_data_summary
)
import json
import io
import psutil
import time
import shutil
import pandas as pd

def get_system_metrics():
    return {
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory_percent': psutil.virtual_memory().percent,
        'memory_available_gb': psutil.virtual_memory().available / 1024 / 1024 / 1024,
        'disk_usage_percent': psutil.disk_usage('/').percent,
    }

@csrf_exempt  # biar gampang testing (nanti bisa pakai token CSRF kalo udah serius)
def analyze_view(request):
    if request.method == "POST":
        # Clean up old PDFs before processing
        cleanup_old_pdfs(hours=0)
        
        if request.content_type == "application/json":
            user_config = json.loads(request.body)
            validation_id = user_config.get("validation_id")
            algorithm = user_config.get("algorithms")[0]
            num_of_cluster = user_config.get("numClusters")
            cities = user_config.get("locations").get("cities")
            commodities = user_config.get("commodities")
            start_year = int(user_config.get("yearRange").get("start"))
            end_year = int(user_config.get("yearRange").get("end"))
            years = list(range(start_year, end_year + 1))
            
            print("\n=== CONFIGURATION ===")
            if user_config:
                try:
                    print(json.dumps(user_config, indent=4))
                except Exception as e:
                    print("Failed parsing configuration:", e)
                    user_config = None
            else:
                user_config = None
                print("No configuration is sent.")
            
        else:
            return JsonResponse({"error": "Unsupported content type"}, status=400)
        
        feature_engineering_config = FeatureEngineeringConfig(
            filter_years=years,
            filter_commodities=commodities,
            filter_cities=cities
        )
        
        feature_engineering_pipeline = FeatureEngineeringPipeline(feature_engineering_config)
        
        df_consolidated = None
        if validation_id:
            parquet_path = Path("temp_data") / f"validation_{validation_id}.parquet"
            
            if not parquet_path.exists():
                return JsonResponse({"error": "Validation data not found"}, status=404)
            
            df_consolidated = pd.read_parquet(parquet_path)
            
            feature_engineering_results = feature_engineering_pipeline.run_full_pipeline(df_consolidated)
        else:
            feature_engineering_results = feature_engineering_pipeline.run_full_pipeline()

        scaled_df = feature_engineering_results.get("scaled_features").get("robust")
        preprocessed_data = feature_engineering_results.get("consolidated")
        
        clustering_config = ClusteringPipelineConfig.for_api_call(
            coordinates_path="data/city_coordinates.json",
            algorithm=algorithm,
            k=num_of_cluster
        )
        
        pipeline = ClusteringAnalysisPipeline(clustering_config)
        response = pipeline.run_single_clustering(
            scaled_data=scaled_df,              
            preprocessed_data=preprocessed_data,  
            algorithm=algorithm,
            k=num_of_cluster,
            return_format="api"
        )
        
        return JsonResponse(response, safe=False)

    return JsonResponse({"error": "Gunakan metode POST"}, status=400)

@csrf_exempt
def download_pdf(request, analysis_id):
    """
    Download PDF report by analysis ID
    GET /api/clustering/download-pdf/<analysis_id>/
    """
    if request.method == "GET":
        try:
            # Validate analysis_id format
            if not analysis_id.startswith("anl_"):
                return JsonResponse({'error': 'Invalid analysis ID format'}, status=400)
            
            # Construct PDF path
            pdf_path = Path("temp_pdfs") / f"analysis_{analysis_id}.pdf"
            
            # Check if PDF exists
            if not pdf_path.exists():
                return JsonResponse({'error': 'PDF not found'}, status=404)
            
            # Return PDF file
            with open(pdf_path, 'rb') as pdf_file:
                response = HttpResponse(
                    pdf_file.read(), 
                    content_type='application/pdf'
                )
                response['Content-Disposition'] = f'attachment; filename="clustering_report_{analysis_id}.pdf"'
                return response
                
        except Exception as e:
            return JsonResponse({'error': f'Error downloading PDF: {str(e)}'}, status=500)
    
    return JsonResponse({"error": "Use GET method"}, status=400)

@csrf_exempt
def validate_data_view(request):
    """
    Validate uploaded ZIP file containing Excel data.
    POST /api/clustering/validate-data/
    """
    # Import services
    from .services import (
        DataValidationService, FileProcessingService, 
        ValidationResponseBuilder, ValidationLogger
    )
    from .exceptions import ValidationError, FileFormatError
    from .constants import MAX_VALIDATION_AGE_HOURS
    
    # Initialize services
    logger = ValidationLogger()
    validation_service = DataValidationService()
    file_processor = FileProcessingService()
    response_builder = ValidationResponseBuilder()
    
    if request.method != "POST":
        return response_builder.build_method_not_allowed_response()
    
    try:
        # Extract and validate file presence
        file = request.FILES.get("file")
        if not file:
            raise FileFormatError("No file provided in request")
        
        logger.log_validation_start(file.name, file.size)
        
        # Clean up old validation files
        files_cleaned = file_processor.cleanup_old_files(hours=MAX_VALIDATION_AGE_HOURS)
        if files_cleaned > 0:
            logger.log_cleanup_operation(files_cleaned, MAX_VALIDATION_AGE_HOURS)
        
        # Run validation pipeline
        validation_result = validation_service.run_all_validations(file)
        
        if not validation_result.is_valid:
            logger.log_validation_error("validation", validation_result.errors[0] if validation_result.errors else "Unknown validation error")
            return response_builder.build_error_response(
                errors=validation_result.errors,
                status_code=400
            )
        
        # Process and save validated data
        validation_id = f"val_{int(time.time())}"
        file_processor.process_zip_file(file, validation_id)
        parquet_path = file_processor.save_validated_data(validation_result.data, validation_id)
        
        logger.log_file_processing("save", parquet_path, True)
        
        # Extract available options
        available_data = extract_available_options(validation_result.data)
        
        logger.log_validation_success(validation_id, validation_result.metrics)
        
        return response_builder.build_success_response(
            validation_id=validation_id,
            available_data=available_data,
            warnings=validation_result.warnings
        )
        
    except ValidationError as e:
        logger.log_validation_error(e.error_type, str(e), e.details)
        return response_builder.build_validation_error_response(e)
    
    except Exception as e:
        logger.log_validation_error("SYSTEM_ERROR", str(e), {})
        return response_builder.build_system_error_response(
            "An unexpected error occurred during validation",
            str(e)
        )
        
def cleanup_old_pdfs(hours: int = 24):
    """
    Clean up PDFs older than specified hours
    """
    try:
        temp_pdfs_dir = Path("temp_pdfs")
        if not temp_pdfs_dir.exists():
            return
        
        cutoff_time = time.time() - (hours * 60 * 60)
        cleaned_count = 0
        
        for pdf_file in temp_pdfs_dir.glob("analysis_*.pdf"):
            if pdf_file.stat().st_mtime < cutoff_time:
                pdf_file.unlink()
                cleaned_count += 1
        
        if cleaned_count > 0:
            print(f"🧹 Cleaned up {cleaned_count} old PDF files")
            
    except Exception as e:
        print(f"❌ Error during cleanup: {e}")


def cleanup_old_validations(hours: int = 1):
    """
    Clean up validation Parquet files older than specified hours
    """
    try:
        temp_data_dir = Path("temp_data")
        if not temp_data_dir.exists():
            return
        
        cutoff_time = time.time() - (hours * 60 * 60)
        cleaned_count = 0
        
        for parquet_file in temp_data_dir.glob("validation_*.parquet"):
            if parquet_file.stat().st_mtime < cutoff_time:
                parquet_file.unlink()
                cleaned_count += 1
        
        if cleaned_count > 0:
            print(f"🧹 Cleaned up {cleaned_count} old validation files")
            
    except Exception as e:
        print(f"❌ Error during validation cleanup: {e}")


