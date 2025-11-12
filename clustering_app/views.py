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
import numpy as np

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
            algorithms = user_config.get("algorithms", [])
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
        
        # Check if this is a comparison request
        if len(algorithms) > 1:
            # COMPARISON MODE
            print(f"Running algorithm comparison with {len(algorithms)} algorithms: {algorithms}")
            
            # Create comparison config
            clustering_config = ClusteringPipelineConfig.for_comparison_call(
                coordinates_path="data/city_coordinates.json",
                algorithms=algorithms,
                k=num_of_cluster
            )
            
            # Run comparison
            return run_algorithm_comparison(
                scaled_df=scaled_df,
                preprocessed_df=preprocessed_data,
                algorithms=algorithms,
                k=num_of_cluster,
                clustering_config=clustering_config
            )
        else:
            # SINGLE ALGORITHM MODE (existing flow)
            algorithm = algorithms[0] if algorithms else "kmeans"
            
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
                pdf_path = Path("temp_pdfs") / f"comparison_{analysis_id}.pdf"
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


def run_algorithm_comparison(
    scaled_df: pd.DataFrame,
    preprocessed_df: pd.DataFrame,
    algorithms: list,
    k: int,
    clustering_config: ClusteringPipelineConfig
) -> JsonResponse:
    """
    Run multiple clustering algorithms and compare their performance.
    
    Args:
        scaled_df: Scaled feature DataFrame
        preprocessed_df: Preprocessed data DataFrame
        algorithms: List of algorithm names to compare
        k: Number of clusters
        clustering_config: Base clustering configuration
        
    Returns:
        JsonResponse with comparison results and PDF path
    """
    comparison_results = {}
    all_results = {}  # Store full results for PDF generation
    
    for algorithm in algorithms:
        try:
            # Create pipeline for this algorithm
            pipeline = ClusteringAnalysisPipeline(clustering_config)
            
            # Run clustering (return_format="detailed" for metrics)
            result = pipeline.run_single_clustering(
                scaled_data=scaled_df,
                preprocessed_data=preprocessed_df,
                algorithm=algorithm,
                k=k,
                return_format="detailed"
            )
            
            # Store full results for PDF
            all_results[algorithm] = result
            
            # Extract cluster assignments
            clusters = []
            cities = []
            for assignment in result["assignments"]:
                clusters.append({
                    "city": assignment["name"],
                    "cluster": assignment["clusterId"]
                })
                cities.append(assignment["name"])
            
            # Extract cluster information and create cluster metadata
            cluster_metadata = []
            unique_clusters = sorted(set(assignment["clusterId"] for assignment in result["assignments"]))
            
            # Default color palette for clusters
            color_palette = [
                {"id": 0, "name": "Klaster 0: Pusat Konsumsi Harga Tinggi", "color": "text-red-500", "bgColor": "bg-red-500", "hexColor": "#EF4444"},
                {"id": 1, "name": "Klaster 1: Lumbung Pangan Stabil", "color": "text-green-500", "bgColor": "bg-green-500", "hexColor": "#22C55E"},
                {"id": 2, "name": "Klaster 2: Wilayah Transisi", "color": "text-blue-500", "bgColor": "bg-blue-500", "hexColor": "#3B82F6"},
                {"id": 3, "name": "Klaster 3: Daerah Perbatasan", "color": "text-purple-500", "bgColor": "bg-purple-500", "hexColor": "#8B5CF6"},
                {"id": 4, "name": "Klaster 4: Zona Khusus", "color": "text-orange-500", "bgColor": "bg-orange-500", "hexColor": "#F97316"},
                {"id": 5, "name": "Klaster 5: Area Strategis", "color": "text-pink-500", "bgColor": "bg-pink-500", "hexColor": "#EC4899"},
                {"id": 6, "name": "Klaster 6: Wilayah Utama", "color": "text-indigo-500", "bgColor": "bg-indigo-500", "hexColor": "#6366F1"},
                {"id": 7, "name": "Klaster 7: Zona Prioritas", "color": "text-teal-500", "bgColor": "bg-teal-500", "hexColor": "#14B8A6"},
                {"id": 8, "name": "Klaster 8: Daerah Khusus", "color": "text-yellow-500", "bgColor": "bg-yellow-500", "hexColor": "#EAB308"},
                {"id": 9, "name": "Klaster 9: Wilayah Terpencil", "color": "text-gray-500", "bgColor": "bg-gray-500", "hexColor": "#6B7280"}
            ]
            
            for cluster_id in unique_clusters:
                if cluster_id < len(color_palette):
                    cluster_metadata.append(color_palette[cluster_id])
                else:
                    # Fallback for additional clusters
                    cluster_metadata.append({
                        "id": cluster_id,
                        "name": f"Klaster {cluster_id}: Cluster {cluster_id}",
                        "color": "text-gray-500",
                        "bgColor": "bg-gray-500",
                        "hexColor": "#6B7280"
                    })
            
            # Build comparison response
            comparison_results[algorithm] = {
                "clusters": cluster_metadata,
                "cities": result["assignments"],  # Keep original assignments with lat/lon
                "silhouette_score": result["metrics"]["silhouette_score"],
                "dbi_score": result["metrics"]["davies_bouldin_index"],
                "computation_time": result["metadata"].get("execution_time_seconds", 0)
            }
            
        except Exception as e:
            print(f"Error running {algorithm}: {e}")
            comparison_results[algorithm] = {
                "error": str(e),
                "clusters": [],
                "cities": [],
                "silhouette_score": 0,
                "dbi_score": float('inf'),
                "computation_time": 0
            }
    
    # Generate comparison PDF
    analysis_id = f"anl_{int(time.time())}"
    temp_pdfs_dir = Path("temp_pdfs")
    temp_pdfs_dir.mkdir(exist_ok=True)
    pdf_path = temp_pdfs_dir / f"comparison_{analysis_id}.pdf"
    
    # Create visualization service for PDF generation
    from src.visualization.pipeline import ClusterVisualizationService
    viz_service = ClusterVisualizationService(
        scaled_df=scaled_df,
        preprocessed_df=preprocessed_df,
        labels=np.array([]),  # Dummy labels for initialization
        model=None,  # Dummy model
        silhouette_avg=0.0,  # Dummy score
        algorithm_name="comparison",
        output_path=temp_pdfs_dir
    )
    
    # Generate comparison PDF
    viz_service.generate_comparison_pdf(
        comparison_results=all_results,
        output_path=pdf_path,
        k=k,
        algorithm_names=algorithms
    )
    
    return JsonResponse({
        "analysis_id": analysis_id,
        "years": [str(year) for year in range(2020, 2025)],  # Default years, could be extracted from data
        "algorithm_results": comparison_results
    }, safe=False)
        
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
        
        for pdf_file in temp_pdfs_dir.glob("*.pdf"):
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


