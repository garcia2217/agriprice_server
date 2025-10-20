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
        cleanup_old_pdfs(hours=24)
        
        before_metrics = get_system_metrics()
        start_time = time.time()
        
        if request.content_type.startswith("multipart/form-data"):
            # Mode upload file
            file = request.FILES.get("file")
            user_config = json.loads(request.POST.get("config", "{}"))
        elif request.content_type == "application/json":
            # Mode JSON only
            data = json.loads(request.body)
            file = None
            user_config = data
            # filter data
            commodities = user_config.get("commodities")
            start_year = int(user_config.get("yearRange").get("start"))
            end_year = int(user_config.get("yearRange").get("end"))
            years = list(range(start_year, end_year + 1))
            cities = user_config.get("locations").get("cities")
        else:
            return JsonResponse({"error": "Unsupported content type"}, status=400)
        
        # Check if this is a validation_id mode
        validation_id = user_config.get("validation_id")
        if validation_id:
            # Load data from Parquet file
            parquet_path = Path("temp_data") / f"validation_{validation_id}.parquet"
            
            if not parquet_path.exists():
                return JsonResponse({"error": "Validation data not found"}, status=404)
            
            consolidated_df = pd.read_parquet(parquet_path)
            
            # Filter based on user selections
            selected_commodities = user_config.get("selected_commodities")
            selected_years = user_config.get("selected_years")
            selected_cities = user_config.get("selected_cities")
            
            # Apply filters to consolidated_df
            if selected_commodities:
                consolidated_df = consolidated_df[consolidated_df["Commodity"].isin(selected_commodities)]
            if selected_years:
                consolidated_df = consolidated_df[consolidated_df["Year"].astype(int).isin(selected_years)]
            if selected_cities:
                consolidated_df = consolidated_df[consolidated_df["City"].isin(selected_cities)]
            
            # Set up feature engineering config for validated data
            feature_engineering_config = FeatureEngineeringConfig()
            file = None  # No file upload in validation mode
        
        print("=== FILE DITERIMA ===")
        print(file)
        print("Nama file:", file.name if file else None)
        print("Ukuran:", file.size if file else None)

        print("\n=== CONFIG ===")
        if user_config:
            try:
                print(json.dumps(user_config, indent=4))
            except Exception as e:
                print("Gagal parse config:", e)
                user_config = None
        else:
            user_config = None
            print("Tidak ada config dikirim")
            
            
        # user config
        algorithm = user_config.get("algorithms")[0]
        num_of_cluster = user_config.get("numClusters")
        
        # Handle different input modes
        if validation_id:
            # Data already loaded from Parquet file above
            pass
        elif file:
            config = ConsolidationConfig(
                input_type="zip",
            )
            
            zip_bytes = file.read()
            zip_stream = io.BytesIO(zip_bytes)
            
            consolidator = DataConsolidator(config)
            results = consolidator.process_zip_stream(
                zip_stream=zip_stream
            )
            
            consolidated_df = results.get("consolidated_df")
            
            if consolidated_df is None:
                print("error consolidation process failed")
        
            feature_engineering_config = FeatureEngineeringConfig()
        else:
            consolidated_df = None
            
            feature_engineering_config = FeatureEngineeringConfig(
                filter_years=years,
                filter_commodities=commodities,
                filter_cities=cities
            )
            
        feature_engineering_pipeline = FeatureEngineeringPipeline(feature_engineering_config)
        feature_engineering_results = feature_engineering_pipeline.run_full_pipeline(
            input_data=consolidated_df
        )
        
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
        
        after_metrics = get_system_metrics()
        end_time = time.time()
        
        # NEW: Calculate and log metrics
        duration = end_time - start_time
        cpu_impact = after_metrics['cpu_percent'] - before_metrics['cpu_percent']
        memory_impact = after_metrics['memory_percent'] - before_metrics['memory_percent']
        
        print(f"=== CLUSTERING ANALYSIS METRICS ===")
        print(f"Duration: {duration:.2f}s")
        print(f"CPU Impact: {cpu_impact:.2f}%")
        print(f"Memory Impact: {memory_impact:.2f}%")
        print(f"Available Memory: {after_metrics['memory_available_gb']:.2f}GB")
        print(f"=================================")

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


@csrf_exempt
def validate_data_view(request):
    """
    Validate uploaded ZIP file containing Excel data.
    POST /api/clustering/validate-data/
    """
    if request.method == "POST":
        # Clean up old validation files before processing
        cleanup_old_validations(hours=1)
        
        try:
            # Extract ZIP file from request
            file = request.FILES.get("file")
            
            # Validate file format
            is_valid_format, format_errors = validate_file_format(file)
            if not is_valid_format:
                return JsonResponse({
                    "valid": False,
                    "errors": format_errors
                }, status=400)
            
            # Generate unique validation ID
            validation_id = f"val_{int(time.time())}"
            
            # Process ZIP file using consolidation pipeline
            config = ConsolidationConfig(input_type="zip")
            consolidator = DataConsolidator(config)
            
            # Reset file position and read bytes
            file.seek(0)
            zip_bytes = file.read()
            zip_stream = io.BytesIO(zip_bytes)
            zip_stream.seek(0)  # Ensure stream is at beginning
            
            results = consolidator.process_zip_stream(zip_stream=zip_stream)
            
            if not results.get("success", False):
                return JsonResponse({
                    "valid": False,
                    "errors": [f"Data consolidation failed: {results.get('error', 'Unknown error')}"]
                }, status=400)
            
            consolidated_df = results.get("consolidated_df")
            if consolidated_df is None:
                return JsonResponse({
                    "valid": False,
                    "errors": ["Failed to consolidate data from ZIP file"]
                }, status=400)
            
            # Run comprehensive validation checks
            all_errors = []
            all_warnings = []
            
            # 1. Geographic scope validation
            valid_cities = config.valid_cities
            is_valid_geo, geo_errors = validate_geographic_scope(consolidated_df, valid_cities)
            if not is_valid_geo:
                all_errors.extend(geo_errors)
            
            # 2. Data structure validation
            is_valid_structure, structure_errors = validate_data_structure(consolidated_df)
            if not is_valid_structure:
                all_errors.extend(structure_errors)
            
            # 3. Commodity validation
            supported_commodities = config.commodities
            is_valid_commodities, commodity_errors = validate_commodities(consolidated_df, supported_commodities)
            if not is_valid_commodities:
                all_errors.extend(commodity_errors)
            
            # 4. Temporal validation
            is_valid_temporal, temporal_errors = validate_temporal_range(consolidated_df)
            if not is_valid_temporal:
                all_errors.extend(temporal_errors)
            
            # 5. Data quality validation
            is_valid_quality, quality_warnings, quality_metrics = validate_data_quality(consolidated_df)
            all_warnings.extend(quality_warnings)
            
            # If any critical validation failed, return errors
            if all_errors:
                return JsonResponse({
                    "valid": False,
                    "errors": all_errors
                }, status=400)
            
            # Save validated data as Parquet
            temp_data_dir = Path("temp_data")
            temp_data_dir.mkdir(exist_ok=True)
            parquet_path = temp_data_dir / f"validation_{validation_id}.parquet"
            consolidated_df.to_parquet(parquet_path)
            
            # Extract available options
            available_data = extract_available_options(consolidated_df)
            
            # Calculate data summary
            data_summary = calculate_data_summary(consolidated_df)
            
            # Return success response
            return JsonResponse({
                "validation_id": validation_id,
                "valid": True,
                "errors": [],
                "warnings": all_warnings,
                "available_data": available_data,
                "data_quality": quality_metrics
            })
            
        except Exception as e:
            return JsonResponse({
                "valid": False,
                "errors": [f"Validation failed: {str(e)}"]
            }, status=500)
    
    return JsonResponse({"error": "Use POST method"}, status=400)
