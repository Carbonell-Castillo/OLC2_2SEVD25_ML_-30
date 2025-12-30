
**Base URL:** `http://localhost:8000`  

---

## Endpoints

### 1. **GET /** - Información General
Endpoint de bienvenida que retorna información básica de la API.

**Respuesta:**
```json
{
  "message": "Bienvenido a InsightCluster API",
  "version": "1.0.0",
  "status": "running",
  "endpoints": {...}
}
```

---

### 2. **POST /upload** - Subir Archivo

Carga un archivo CSV o Excel para análisis de clustering.

**Request:**
- **Content-Type:** `multipart/form-data`
- **Parámetro:** `file` (archivo CSV o Excel)

**Columnas requeridas en el archivo:**
- `cliente_id`
- `frecuencia_compra`
- `monto_total_gastado`
- `canal_principal`

**Respuesta exitosa:**
```json
{
  "status": "success",
  "message": "Archivo cargado correctamente",
  "file_id": "20250128_143022",
  "rows": 500,
  "columns": ["cliente_id", "frecuencia_compra", ...]
}
```

**Errores:**
- `400`: Formato no soportado o columnas faltantes
- `500`: Error interno al procesar archivo

---

### 3. **GET /info/{file_id}** - Información del Archivo

Obtiene metadatos de un archivo previamente cargado.

**Parámetros:**
- `file_id` (path): ID único del archivo

**Respuesta:**
```json
{
  "filename": "clientes.csv",
  "filepath": "data/uploads/20250128_143022.csv",
  "rows": 500,
  "columns": ["cliente_id", ...],
  "uploaded_at": "2025-01-28T14:30:22"
}
```

**Error:**
- `404`: Archivo no encontrado

---

### 4. **POST /train** - Entrenar Modelo

Entrena el modelo K-Means con el archivo especificado.

**Request Body:**
```json
{
  "file_id": "20250128_143022",
  "n_clusters": 3,
  "max_iterations": 100,
  "random_state": 42
}
```

**Parámetros:**
- `file_id` (requerido): ID del archivo a procesar
- `n_clusters` (opcional, default: 3): Número de clusters
- `max_iterations` (opcional, default: 100): Iteraciones máximas
- `random_state` (opcional, default: 42): Semilla aleatoria

**Respuesta exitosa:**
```json
{
  "status": "success",
  "file_id": "20250128_143022",
  "summary": {
    "n_clusters": 3,
    "overall_quality": "Buena",
    "recommendation": "Clusters bien definidos"
  },
  "metrics": {
    "silhouette_score": 0.6234,
    "davies_bouldin_score": 0.7821
  },
  "clusters": [
    {
      "cluster_id": 0,
      "size": 150,
      "percentage": 30.0,
      "description": "Clientes de alto valor",
      "main_channel": "web"
    }
  ],
  "reviews": {
    "enabled": true,
    "clusters": [...]
  }
}
```

**Errores:**
- `404`: File ID no encontrado
- `500`: Error durante entrenamiento

---

### 5. **GET /results/{file_id}** - Obtener Resultados

Recupera los resultados de un modelo previamente entrenado.

**Parámetros:**
- `file_id` (path): ID del archivo procesado

**Respuesta:** Misma estructura que `/train`

**Error:**
- `404`: Resultados no encontrados (archivo no entrenado)

---

### 6. **GET /download/{file_id}** - Descargar CSV

Descarga el archivo CSV con los resultados del clustering (columna `cluster` agregada).

**Parámetros:**
- `file_id` (path): ID del archivo procesado

**Respuesta:**
- **Content-Type:** `text/csv`
- **Archivo:** `resultados_{file_id}.csv`

**Errores:**
- `404`: Archivo no encontrado o no procesado

---

### 7. **GET /metrics/info** - Información de Métricas

Retorna rangos e interpretación de las métricas de evaluación.

**Respuesta:**
```json
{
  "silhouette_score": {
    "range": [-1, 1],
    "interpretation": "Valores cercanos a 1 indican clusters bien separados",
    "good_threshold": 0.5
  },
  "davies_bouldin_score": {...}
}
```

---

### 8. **GET /visualization/{file_id}** - Generar Visualización PCA

Genera una visualización 2D mediante PCA de los clusters identificados.

**Parámetros:**
- `file_id` (path): ID del archivo procesado

**Respuesta:**
```json
{
  "status": "success",
  "file_id": "20250128_143022",
  "image_path": "data/visualizations/20250128_143022_pcaGrafica.png",
  "cluster_names": {
    "0": "Clientes Digitales",
    "1": "Clientes Presenciales",
    "2": "Clientes Telefónicos"
  },
  "explained_variance": {
    "pc1": 0.6523,
    "pc2": 0.2134,
    "total": 0.8657
  }
}
```

**Nombres de clusters inferidos:**
- **Clientes Digitales**: canal principal "web"
- **Clientes Presenciales**: canal principal "tienda física"
- **Clientes Telefónicos**: canal principal "call center"
- **Clientes Mixtos**: otros casos

**Errores:**
- `404`: Resultados no encontrados
- `400`: Clusters no disponibles

---

### 9. **GET /export/pdf/{file_id}** - Exportar Reporte PDF

Genera un reporte PDF completo con métricas, clusters y visualización PCA.

**Parámetros:**
- `file_id` (path): ID del archivo procesado

**Respuesta:**
```json
{
  "status": "success",
  "message": "Reporte PDF generado correctamente",
  "pdf_path": "data/reports/20250128_143022_reporte.pdf",
  "includes_visualization": true
}
```

**Contenido del PDF:**
1. Resumen del modelo (número de clusters, calidad, recomendación)
2. Métricas de evaluación (Silhouette, Davies-Bouldin)
3. Segmentos identificados (tamaño, canal, descripción)
4. Visualización PCA (si está disponible)

**Error:**
- `404`: Resultados no encontrados

---

## Flujo de Trabajo Recomendado

```
1. POST /upload          → Subir archivo CSV/Excel
2. POST /train           → Entrenar modelo K-Means
3. GET /results/{id}     → Consultar resultados
4. GET /visualization/{id} → Generar gráfica PCA
5. GET /export/pdf/{id}  → Descargar reporte completo
6. GET /download/{id}    → Descargar CSV con clusters
```

---

## Notas Técnicas

### Preprocesamiento
- Limpieza automática de datos
- Normalización con StandardScaler
- Encoding de variables categóricas

### Clustering de Texto (Opcional)
Si el CSV incluye columna `texto_limpio`, se realiza clustering de reseñas usando TF-IDF.

### Almacenamiento
Los datos se guardan temporalmente en memoria (`storage` dict) y en disco:
- `data/uploads/` - Archivos originales
- `data/exports/` - CSVs con clusters
- `data/visualizations/` - Gráficas PCA
- `data/reports/` - Reportes PDF

### CORS
Configurado para aceptar peticiones desde cualquier origen (`allow_origins=["*"]`).