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
import json
import io
import psutil
import time
import shutil

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
        
        consolidated_df = None
        if file:
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
