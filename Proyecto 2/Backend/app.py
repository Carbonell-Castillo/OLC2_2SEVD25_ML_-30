# backend/app.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
from datetime import datetime
import os

from models.preprocessing import DataPreprocessor
from models.kmeans import KMeans
from models.text_clustering import TextClusterer
from utils.metrics import ClusteringMetrics
from models.profiling import ClusterProfiler


app = FastAPI(
    title="InsightCluster API",
    description="API para clustering de clientes con K-Means",
    version="1.0.0"
)

# Configurar CORS (permite conexión desde frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Almacenamiento temporal
storage = {}

#  MODELO DE DATOS PARA TRAINING 
class TrainRequest(BaseModel):
    """Modelo de datos para request de entrenamiento"""
    file_id: str
    n_clusters: int = 3
    max_iterations: int = 100
    random_state: int = 42

@app.get("/")
def root():
    """Endpoint de bienvenida"""
    return {
        "message": "Bienvenido a InsightCluster API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "docs": "/docs",
            "upload": "/upload",
            "train": "/train",
            "results": "/results/{file_id}",
            "download": "/download/{file_id}"
        }
    }

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Subir archivo CSV o Excel para análisis
    """
    try:
        # Validar formato
        filename = file.filename
        if not (filename.endswith('.csv') or filename.endswith(('.xlsx', '.xls'))):
            raise HTTPException(
                status_code=400,
                detail="Formato no soportado. Use CSV o Excel (.xlsx, .xls)"
            )
        
        # Leer archivo
        if filename.endswith('.csv'):
            df = pd.read_csv(file.file)
        else:
            df = pd.read_excel(file.file)
        
        # Validar columnas requeridas
        required_cols = [
            'cliente_id', 'frecuencia_compra', 'monto_total_gastado',
            'canal_principal'
        ]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Faltan columnas requeridas: {missing_cols}"
            )
        
        # Generar ID único
        file_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Guardar archivo
        filepath = f"data/uploads/{file_id}.csv"
        df.to_csv(filepath, index=False)
        
        # Almacenar información
        storage[file_id] = {
            "filename": filename,
            "filepath": filepath,
            "rows": len(df),
            "columns": list(df.columns),
            "uploaded_at": datetime.now().isoformat()
        }
        
        return {
            "status": "success",
            "message": "Archivo cargado correctamente",
            "file_id": file_id,
            "rows": len(df),
            "columns": list(df.columns)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/info/{file_id}")
def get_file_info(file_id: str):
    """Obtener información de un archivo cargado"""
    if file_id not in storage:
        raise HTTPException(status_code=404, detail="Archivo no encontrado")
    
    return storage[file_id]

@app.post("/train")
def train_model(request: TrainRequest):
    """Entrenar modelo K-Means y devolver respuesta limpia para frontend"""

    file_id = request.file_id
    n_clusters = request.n_clusters
    max_iterations = request.max_iterations
    random_state = request.random_state

    # ================= VALIDAR FILE =================
    if file_id not in storage:
        raise HTTPException(status_code=404, detail="File ID not found")

    file_path = storage[file_id]["filepath"]

    # ================= PREPROCESAMIENTO =================
    preprocessor = DataPreprocessor()
    X_scaled, df, feature_names = preprocessor.load_and_clean(file_path)

    # ================= K-MEANS CLIENTES =================
    kmeans = KMeans(
        n_clusters=n_clusters,
        max_iterations=max_iterations,
        random_state=random_state
    )
    kmeans.fit(X_scaled)

    df["cluster"] = kmeans.labels

    # ================= STATS =================
    cluster_stats = preprocessor.calculate_cluster_stats(df, kmeans.labels)

    # ================= MÉTRICAS =================
    metrics = ClusteringMetrics.calculate_all_metrics(X_scaled, kmeans.labels)

    # ================= CLUSTERING DE TEXTO =================
    review_clustering = None

    if "texto_limpio" in df.columns:
        try:
            text_clusterer = TextClusterer(
                n_clusters=3,
                max_features=100
            )

            review_labels = text_clusterer.fit_predict(df["texto_limpio"])
            top_terms = text_clusterer.get_top_terms(n_terms=5)
            review_stats = text_clusterer.get_cluster_stats(review_labels)

            review_clustering = {
                "n_clusters": 3,
                "cluster_stats": review_stats,
                "top_keywords": top_terms
            }

        except Exception:
            review_clustering = None

    # ================= PROFILING =================
    profiler = ClusterProfiler(
        df=df,
        cluster_stats=cluster_stats,
        review_clustering=review_clustering
    )

    cluster_profiles = profiler.generate_profiles()

    # ================= EXPORT CSV =================
    output_path = f"data/exports/{file_id}_clustered.csv"
    df.to_csv(output_path, index=False)

    # ================= GUARDAR RESULTADOS COMPLETOS =================
    storage[file_id]["results"] = {
        "n_clusters": n_clusters,
        "inertia": kmeans.inertia_,
        "cluster_stats": cluster_stats,
        "cluster_profiles": cluster_profiles,
        "metrics": metrics,
        "feature_names": feature_names,
        "review_clustering": review_clustering,
        "output_path": output_path,
        "trained_at": datetime.now().isoformat()
    }

    # ================= RESPUESTA LIMPIA PARA FRONTEND =================
    clean_response = {
        "status": "success",
        "file_id": file_id,
        "summary": {
            "n_clusters": n_clusters,
            "overall_quality": metrics.get("interpretation", {}).get("overall_quality"),
            "recommendation": metrics.get("interpretation", {}).get("recommendation")
        },
        "metrics": {
            "silhouette_score": metrics.get("silhouette_score"),
            "davies_bouldin_score": metrics.get("davies_bouldin_score")
        },
        "clusters": [],
        "reviews": {
            "enabled": review_clustering is not None,
            "clusters": []
        }
    }

    # Agregar perfiles de clusters
    for profile in cluster_profiles:
        clean_response["clusters"].append({
            "cluster_id": profile["cluster_id"],
            "size": profile["size"],
            "percentage": profile["percentage"],
            "description": profile["description"],
            "main_channel": profile.get("canal_principal")
        })

    # Agregar info de reseñas
    if review_clustering:
        for cluster_id, keywords in review_clustering["top_keywords"].items():
            clean_response["reviews"]["clusters"].append({
                "cluster_id": cluster_id,
                "top_keywords": keywords
            })

    return clean_response

@app.get("/results/{file_id}")
def get_results(file_id: str):
    """Obtener resultados del clustering en formato limpio para frontend"""

    if file_id not in storage or "results" not in storage[file_id]:
        raise HTTPException(status_code=404, detail="Resultados no encontrados")

    results = storage[file_id]["results"]
    metrics = results.get("metrics", {})
    cluster_profiles = results.get("cluster_profiles", [])
    review_clustering = results.get("review_clustering")

    clean_response = {
        "status": "success",
        "file_id": file_id,
        "summary": {
            "n_clusters": results.get("n_clusters"),
            "overall_quality": metrics.get("interpretation", {}).get("overall_quality"),
            "recommendation": metrics.get("interpretation", {}).get("recommendation"),
            "trained_at": results.get("trained_at")
        },
        "metrics": {
            "silhouette_score": metrics.get("silhouette_score"),
            "davies_bouldin_score": metrics.get("davies_bouldin_score")
        },
        "clusters": [],
        "reviews": {
            "enabled": review_clustering is not None,
            "clusters": []
        }
    }

    # Agregar perfiles de clusters
    for profile in cluster_profiles:
        clean_response["clusters"].append({
            "cluster_id": profile["cluster_id"],
            "size": profile["size"],
            "percentage": profile["percentage"],
            "description": profile["description"],
            "main_channel": profile.get("canal_principal")
        })

    # Agregar info de reseñas (si existe)
    if review_clustering:
        for cluster_id, keywords in review_clustering.get("top_keywords", {}).items():
            clean_response["reviews"]["clusters"].append({
                "cluster_id": cluster_id,
                "top_keywords": keywords
            })

    return clean_response


@app.get("/download/{file_id}")
def download_results(file_id: str):
    """Descargar CSV con resultados"""
    if file_id not in storage:
        raise HTTPException(404, "Archivo no encontrado")
    
    if "results" not in storage[file_id]:
        raise HTTPException(404, "Este archivo no ha sido procesado")
    
    filepath = storage[file_id]["results"]["output_path"]
    
    if not os.path.exists(filepath):
        raise HTTPException(404, "Archivo de resultados no encontrado")
    
    return FileResponse(
        filepath,
        media_type='text/csv',
        filename=f"resultados_{file_id}.csv"
    )

@app.get("/metrics/info")
def get_metrics_info():
    """
    Obtener información sobre las métricas de evaluación
    """
    return ClusteringMetrics.get_metric_ranges()


@app.get("/health")
def health_check():
    """Verificar estado de la API"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "files_uploaded": len(storage),
        "models_trained": len([f for f in storage.values() if "results" in f])
    }