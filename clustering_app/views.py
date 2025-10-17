from django.shortcuts import render
from django.http import HttpResponse

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

# buat dummy data yang mirip userResults lo (disingkat biar ga kepanjangan)
USER_RESULTS = {
    "cities": [
        {"name": "Kota A", "lat": -1.5, "lon": 102.5, "clusterId": 0},
        {"name": "Kota B", "lat": -2.5, "lon": 106.5, "clusterId": 0},
        {"name": "Kota C", "lat": -4.5, "lon": 110.5, "clusterId": 1},
    ],
    "clusters": [
        {
            "id": 0,
            "name": "Klaster Pengguna 0: Harga Rendah",
            "color": "text-blue-500",
            "bgColor": "bg-blue-500",
            "hexColor": "#3B82F6",
        },
        {
            "id": 1,
            "name": "Klaster Pengguna 1: Harga Sedang",
            "color": "text-purple-500",
            "bgColor": "bg-purple-500",
            "hexColor": "#8B5CF6",
        },
    ],
    "trends": {
        "Beras": [
            {"clusterId": 0, "data": [11000, 11200, 11300]},
            {"clusterId": 1, "data": [13000, 13100, 13200]},
        ]
    },
    "years": ["Thn 1", "Thn 2", "Thn 3"],
}

@csrf_exempt  # biar gampang testing (nanti bisa pakai token CSRF kalo udah serius)
def analyze_view(request):
    if request.method == "POST":
        # ambil file dari form-data
        # uploaded_file = request.FILES.get("file")
        # config_raw = request.POST.get("config")
            
        if request.content_type.startswith("multipart/form-data"):
            # Mode upload file
            file = request.FILES.get("file")
            user_config = json.loads(request.POST.get("config", "{}"))
            mode = "upload"
        elif request.content_type == "application/json":
            # Mode JSON only
            data = json.loads(request.body)
            file = None
            user_config = data
            mode = "json"
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
            # preprocessing data
            config = ConsolidationConfig(
                input_type="zip",
                # provinces=["Jawa Barat"],
                # years=["2020"],
                # commodities=commodities
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
        
        # print(json.dumps(feature_engineering_results, indent=4))
        scaled_df = feature_engineering_results.get("scaled_features").get("robust")
        preprocessed_data = feature_engineering_results.get("consolidated")
        
        clustering_config = ClusteringPipelineConfig.for_api_call(
            coordinates_path="data/city_coordinates.json",
            algorithm=algorithm,
            k=num_of_cluster
        )
        pipeline = ClusteringAnalysisPipeline(clustering_config)

        response = pipeline.run_single_clustering(
            scaled_data=scaled_df,              # City + numeric features
            preprocessed_data=preprocessed_data,  # City, Commodity, Date, Price
            algorithm=algorithm,
            k=num_of_cluster,
            return_format="api"
        )

        # kirim dummy response ke frontend
        return JsonResponse(response, safe=False)

    return JsonResponse({"error": "Gunakan metode POST"}, status=400)
