# backend/app.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib import colors
from PIL import Image

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
# 1. Obtener la ruta absoluta a la carpeta de imágenes
current_dir = os.path.dirname(os.path.abspath(__file__))
visualizations_path = os.path.join(current_dir, "data", "visualizations")

# 2. Montar la carpeta como recursos estáticos
# Esto mapea la URL /visualizations a la carpeta física
app.mount("/static", StaticFiles(directory=visualizations_path), name="static")

#Ahora pra reports
reports_path = os.path.join(current_dir, "data", "reports")
app.mount("/reports", StaticFiles(directory=reports_path), name="reports")
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

@app.get("/visualization/{file_id}")
def generate_visualization(file_id: str):
    """
    Generar visualización PCA 2D optimizada (sin reprocesar datos)
    """

    # ================= VALIDAR =================
    if file_id not in storage or "results" not in storage[file_id]:
        raise HTTPException(status_code=404, detail="Resultados no encontrados")

    results = storage[file_id]["results"]
    clustered_path = results["output_path"]

    if not os.path.exists(clustered_path):
        raise HTTPException(status_code=404, detail="Archivo clusterizado no encontrado")

    # ================= CARGAR CSV CLUSTERIZADO =================
    df = pd.read_csv(clustered_path)

    if "cluster" not in df.columns:
        raise HTTPException(status_code=400, detail="Clusters no disponibles")

    # ================= FEATURES (SIN REPROCESAR) =================
    numeric_cols = [
        "frecuencia_compra",
        "monto_total_gastado",
        "monto_promedio_compra",
        "dias_desde_ultima_compra",
        "antiguedad_cliente_meses",
        "numero_productos_distintos"
    ]

    # Obtener todas las columnas dummy que existan
    dummy_cols = [c for c in df.columns if c.startswith("canal_")]
    
    # Combinar features
    features = numeric_cols + dummy_cols
    
    # CRÍTICO: Verificar que todas las features existen en el dataframe
    available_features = [f for f in features if f in df.columns]
    
    if not available_features:
        raise HTTPException(
            status_code=400, 
            detail="No se encontraron features válidas para la visualización"
        )
    
    # SOLUCIÓN: Seleccionar SOLO las columnas que existen y son numéricas
    X = df[available_features].select_dtypes(include=[np.number]).values
    
    # Verificar que tenemos datos válidos
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise HTTPException(
            status_code=400,
            detail="No hay datos numéricos válidos para la visualización"
        )

    # ================= PCA =================
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    df["pca_1"] = X_pca[:, 0]
    df["pca_2"] = X_pca[:, 1]

    # ================= NOMBRE DE CLUSTER =================
    def infer_cluster_name(cluster_df):
        if "canal_principal" not in cluster_df.columns:
            return "Clientes Mixtos"

        canal = cluster_df["canal_principal"].mode()
        if len(canal) == 0:
            return "Clientes Mixtos"
            
        canal = canal.iloc[0]

        if canal == "web":
            return "Clientes Digitales"
        if canal == "tienda física":
            return "Clientes Presenciales"
        if canal == "call center":
            return "Clientes Telefónicos"

        return "Clientes Mixtos"

    # ================= GRAFICAR =================
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(12, 8), facecolor="white")

    palette = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4B51']
    cluster_names = {}

    for idx, cluster_id in enumerate(sorted(df["cluster"].unique())):
        subset = df[df["cluster"] == cluster_id]
        name = infer_cluster_name(subset)
        cluster_names[int(cluster_id)] = name

        ax.scatter(
            subset["pca_1"],
            subset["pca_2"],
            label=name,
            alpha=0.7,
            s=80,
            color=palette[idx % len(palette)],
            edgecolors="white",
            linewidth=0.5
        )

    # ================= ESTILO =================
    var_pc1 = pca.explained_variance_ratio_[0] * 100
    var_pc2 = pca.explained_variance_ratio_[1] * 100

    ax.set_xlabel(f"PC1 ({var_pc1:.1f}% varianza)", fontsize=12, fontweight="bold")
    ax.set_ylabel(f"PC2 ({var_pc2:.1f}% varianza)", fontsize=12, fontweight="bold")
    ax.set_title("Segmentación de Clientes mediante PCA", fontsize=16, fontweight="bold")

    ax.legend(title="Segmentos")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # ================= GUARDAR =================
    os.makedirs("data/visualizations", exist_ok=True)
    image_path = f"data/visualizations/{file_id}_pcaGrafica.png"
    plt.savefig(image_path, dpi=300, bbox_inches="tight")
    plt.close()

    # ================= RESPUESTA =================
    return {
        "status": "success",
        "file_id": file_id,
        "image_path": image_path,
        "cluster_names": cluster_names,
        "explained_variance": {
            "pc1": round(pca.explained_variance_ratio_[0], 4),
            "pc2": round(pca.explained_variance_ratio_[1], 4),
            "total": round(sum(pca.explained_variance_ratio_), 4)
        }
    }

@app.get("/export/pdf/{file_id}")
def export_pdf(file_id: str):
    """
    Exportar reporte PDF profesional con resultados y visualización PCA
    """

    # ================= VALIDAR =================
    if file_id not in storage or "results" not in storage[file_id]:
        raise HTTPException(status_code=404, detail="Resultados no encontrados")

    results = storage[file_id]["results"]

    # ================= RUTAS =================
    os.makedirs("data/reports", exist_ok=True)
    pdf_path = f"data/reports/{file_id}_reporte.pdf"

    image_path = f"data/visualizations/{file_id}_pcaGrafica.png"
    has_image = os.path.exists(image_path)

    # ================= CREAR PDF =================
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4
    
    # Colores corporativos
    color_primary = colors.HexColor("#2E86AB")
    color_secondary = colors.HexColor("#A23B72")
    color_dark = colors.HexColor("#1a1a1a")
    color_light_gray = colors.HexColor("#f5f5f5")
    
    # ================= PÁGINA 1: PORTADA =================
    # Fondo del encabezado
    c.setFillColor(color_primary)
    c.rect(0, height - 8*cm, width, 8*cm, fill=True, stroke=False)
    
    # Título principal
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 28)
    c.drawCentredString(width/2, height - 4*cm, "InsightCluster")
    
    c.setFont("Helvetica", 16)
    c.drawCentredString(width/2, height - 5*cm, "Reporte de Segmentación de Clientes")
    
    # Información del análisis
    y = height - 10*cm
    c.setFillColor(color_dark)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(3*cm, y, "Información del Análisis")
    y -= 0.8*cm
    
    c.setFont("Helvetica", 10)
    c.drawString(3*cm, y, f"ID de análisis:")
    c.setFont("Helvetica-Bold", 10)
    c.drawString(7*cm, y, f"{file_id}")
    y -= 0.6*cm
    
    c.setFont("Helvetica", 10)
    c.drawString(3*cm, y, f"Fecha de entrenamiento:")
    c.setFont("Helvetica-Bold", 10)
    c.drawString(7*cm, y, f"{results['trained_at']}")
    y -= 0.6*cm
    
    c.setFont("Helvetica", 10)
    c.drawString(3*cm, y, f"Número de clusters:")
    c.setFont("Helvetica-Bold", 10)
    c.drawString(7*cm, y, f"{results['n_clusters']}")
    y -= 1.5*cm
    
    # Resumen ejecutivo
    interpretation = results["metrics"].get("interpretation", {})
    
    if interpretation:
        # Caja con fondo
        c.setFillColor(color_light_gray)
        c.roundRect(2.5*cm, y - 3*cm, width - 5*cm, 3.5*cm, 10, fill=True, stroke=False)
        
        c.setFillColor(color_dark)
        c.setFont("Helvetica-Bold", 12)
        c.drawString(3*cm, y, "Resumen Ejecutivo")
        y -= 0.8*cm
        
        c.setFont("Helvetica", 10)
        c.drawString(3*cm, y, f"Calidad del modelo:")
        c.setFont("Helvetica-Bold", 10)
        c.drawString(7*cm, y, f"{interpretation.get('overall_quality', 'N/A')}")
        y -= 0.6*cm
        
        c.setFont("Helvetica", 10)
        c.drawString(3*cm, y, f"Recomendación:")
        y -= 0.5*cm
        
        # Texto de recomendación con wrap
        recommendation_text = interpretation.get('recommendation', 'N/A')
        c.setFont("Helvetica", 9)
        text_object = c.beginText(3*cm, y)
        text_object.setFont("Helvetica", 9)
        
        # Simple text wrap
        words = recommendation_text.split()
        line = ""
        for word in words:
            if len(line + word) < 70:
                line += word + " "
            else:
                text_object.textLine(line)
                line = word + " "
        if line:
            text_object.textLine(line)
        
        c.drawText(text_object)
    
    # Pie de página
    c.setFillColor(colors.grey)
    c.setFont("Helvetica", 8)
    c.drawCentredString(width/2, 2*cm, "Generado automáticamente por InsightCluster")
    c.drawCentredString(width/2, 1.5*cm, f"Página 1")
    
    # ================= PÁGINA 2: VISUALIZACIÓN PCA =================
    c.showPage()
    y = height - 2.5*cm
    
    # Encabezado
    c.setFillColor(color_primary)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(2*cm, y, "Visualización de Segmentos")
    y -= 0.5*cm
    
    c.setFillColor(color_dark)
    c.setFont("Helvetica", 10)
    c.drawString(2*cm, y, "Análisis de Componentes Principales (PCA)")
    y -= 1.5*cm
    
    # Insertar imagen
    if has_image:
        try:
            # Convertir imagen a RGB
            rgb_image_path = image_path.replace(".png", "_rgb.png")
            
            with Image.open(image_path) as img:
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                img.save(rgb_image_path)
            
            # Dimensiones de la imagen
            img_width = width - 4*cm
            img_height = 16*cm
            img_y = y - img_height
            
            c.drawImage(
                rgb_image_path,
                2*cm,
                img_y,
                width=img_width,
                height=img_height,
                preserveAspectRatio=True,
                mask='auto'
            )
            
            # Limpiar archivo temporal
            if os.path.exists(rgb_image_path) and rgb_image_path != image_path:
                os.remove(rgb_image_path)
            
            y = img_y - 1*cm
            
        except Exception as e:
            c.setFont("Helvetica", 10)
            c.drawString(2*cm, y, f"⚠ Error al cargar visualización: {str(e)}")
            y -= 1*cm
    else:
        c.setFont("Helvetica-Oblique", 10)
        c.drawString(2*cm, y, "Visualización no disponible. Genere primero la gráfica PCA.")
        y -= 1*cm
    
    # Pie de página
    c.setFillColor(colors.grey)
    c.setFont("Helvetica", 8)
    c.drawCentredString(width/2, 1.5*cm, f"Página 2")
    
    # ================= PÁGINA 3: MÉTRICAS Y CLUSTERS =================
    c.showPage()
    y = height - 2.5*cm
    
    # Encabezado
    c.setFillColor(color_primary)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(2*cm, y, "Métricas y Segmentos")
    y -= 1.5*cm
    
    # Métricas en caja
    c.setFillColor(color_light_gray)
    c.roundRect(2*cm, y - 3*cm, width - 4*cm, 3.5*cm, 10, fill=True, stroke=False)
    
    c.setFillColor(color_dark)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2.5*cm, y, "Métricas de Evaluación")
    y -= 0.8*cm
    
    metrics = results["metrics"]
    c.setFont("Helvetica", 10)
    c.drawString(2.5*cm, y, f"Silhouette Score:")
    c.setFont("Helvetica-Bold", 10)
    silhouette = metrics.get('silhouette_score', 0)
    c.drawString(7*cm, y, f"{silhouette:.4f}")
    c.setFont("Helvetica-Oblique", 8)
    c.drawString(10*cm, y, "(Cercano a 1 = mejor separación)")
    y -= 0.6*cm
    
    c.setFont("Helvetica", 10)
    c.drawString(2.5*cm, y, f"Davies-Bouldin Index:")
    c.setFont("Helvetica-Bold", 10)
    davies = metrics.get('davies_bouldin_score', 0)
    c.drawString(7*cm, y, f"{davies:.4f}")
    c.setFont("Helvetica-Oblique", 8)
    c.drawString(10*cm, y, "(Cercano a 0 = mejor separación)")
    y -= 1.8*cm
    
    # Segmentos identificados
    c.setFillColor(color_secondary)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(2*cm, y, "Segmentos Identificados")
    y -= 1*cm
    
    c.setFillColor(color_dark)
    
    for idx, cluster in enumerate(results["cluster_stats"]):
        # Caja para cada cluster
        box_height = 3.2*cm
        
        if y - box_height < 3*cm:
            c.showPage()
            y = height - 2.5*cm
            c.setFillColor(color_dark)
        
        c.setFillColor(color_light_gray)
        c.roundRect(2*cm, y - box_height, width - 4*cm, box_height, 8, fill=True, stroke=False)
        
        c.setFillColor(colors.white)
        c.setFillColor(color_primary)
        c.circle(2.7*cm, y - 0.5*cm, 0.3*cm, fill=True, stroke=False)
        
        c.setFillColor(color_dark)
        c.setFont("Helvetica-Bold", 11)
        c.drawString(3.3*cm, y - 0.65*cm, f"Cluster {cluster['cluster_id']}")
        y -= 1.2*cm
        
        c.setFont("Helvetica", 9)
        c.drawString(2.7*cm, y, f"Tamaño:")
        c.setFont("Helvetica-Bold", 9)
        c.drawString(5*cm, y, f"{cluster['size']} clientes ({cluster['percentage']}%)")
        y -= 0.5*cm
        
        c.setFont("Helvetica", 9)
        c.drawString(2.7*cm, y, f"Canal principal:")
        c.setFont("Helvetica-Bold", 9)
        c.drawString(5*cm, y, f"{cluster.get('main_channel', 'N/A')}")
        y -= 0.5*cm
        
        c.setFont("Helvetica", 9)
        c.drawString(2.7*cm, y, f"Descripción:")
        y -= 0.4*cm
        
        # Wrap description
        description = cluster.get('description', 'Sin descripción')
        c.setFont("Helvetica", 8)
        text_object = c.beginText(2.7*cm, y)
        text_object.setFont("Helvetica", 8)
        
        words = description.split()
        line = ""
        for word in words:
            if len(line + word) < 75:
                line += word + " "
            else:
                text_object.textLine(line)
                line = word + " "
        if line:
            text_object.textLine(line)
        
        c.drawText(text_object)
        y -= 1.3*cm
    
    # Pie de página
    c.setFillColor(colors.grey)
    c.setFont("Helvetica", 8)
    c.drawCentredString(width/2, 1.5*cm, f"Página 3")
    
    # ================= GUARDAR =================
    c.save()

    return {
        "status": "success",
        "message": "Reporte PDF generado correctamente",
        "pdf_path": pdf_path,
        "includes_visualization": has_image,
        "pages": 3
    }


@app.get("/health")
def health_check():
    """Verificar estado de la API"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "files_uploaded": len(storage),
        "models_trained": len([f for f in storage.values() if "results" in f])
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)