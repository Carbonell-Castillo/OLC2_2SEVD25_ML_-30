# InsightCluster - Manual Técnico

**Proyecto 2 - Organización de Lenguajes y Compiladores 2**  
Universidad de San Carlos de Guatemala  
Vacaciones del Segundo Semestre 2025



## 1. Justificación del Enfoque y Modelo Seleccionado

### 1.1 Selección de K-Means

El proyecto requiere segmentación de clientes mediante clustering no supervisado. Se seleccionó K-Means por:

| Criterio | Justificación |
|----------|---------------|
| Complejidad | O(n·k·i) - escalable para 1000+ clientes |
| Interpretabilidad | Centroides representan cliente promedio |
| Implementación | Simple, pocos hiperparámetros |
| Convergencia | Garantizada a mínimo local |
| Industrial | Estándar en segmentación de clientes |

**Alternativas descartadas:**

- **DBSCAN**: Requiere densidad uniforme; datos con variables categóricas heterogéneas
- **Hierarchical**: Complejidad O(n²) prohibitiva
- **GMM**: Asume distribuciones gaussianas que los datos no cumplen

### 1.2 Enfoque Dual

1. **Clustering de Clientes (Numérico)**: Comportamiento de compra
2. **Clustering de Reseñas (Texto)**: TF-IDF + K-Means para sentimiento

---

## 2. Explicación del Preprocesamiento y Decisiones Tomadas

### 2.1 Limpieza de Datos

**Estrategia híbrida:**

| Tipo de Columna | Estrategia | Justificación |
|----------------|------------|---------------|
| Críticas | Eliminar fila | Evita datos inventados |
| Opcionales | Rellenar mediana | Preserva datos |
```python
df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=['frecuencia_compra'])
df['dias_desde_ultima_compra'].fillna(median)
```

**Resultado:** 954/1000 clientes (95.4% retención)

### 2.2 Normalización

**StandardScaler:** media=0, desviación=1
```python
X_scaled = (X - mean) / std
```

**Justificación:** Contribución equitativa de todas las variables

### 2.3 Encoding Categórico

**One-Hot Encoding** para `canal_principal`
```
canal_principal → canal_web, canal_tiendafisica, canal_callcenter
```

### 2.4 Preprocesamiento de Texto

**Pipeline:**
1. Minúsculas
2. Eliminar puntuación/números
3. Remover stopwords
4. Filtrar palabras < 3 caracteres

**TF-IDF:**
```python
TfidfVectorizer(
    max_features=100,
    min_df=2,
    max_df=0.8,
    ngram_range=(1, 2)
)
```

**Resultado:** Matriz (954 reseñas × 100 features)

---

## 3. Justificación de los Hiperparámetros Seleccionados

### 3.1 Número de Clusters (K)

**Evaluación K=3 vs K=5:**

| Métrica | K=3 | K=5 |
|---------|-----|-----|
| Inercia | 8100.56 | 6756.68 |
| Silhouette | 0.1863 | 0.1899 |
| Calinski-Harabasz | 140.49 | 131.23 |
| Davies-Bouldin | 1.9304 | 2.0170 |

**Decisión:** K=3

**Justificación:**
1. Mejor balance de métricas
2. Mayor interpretabilidad (3 segmentos más manejables)
3. Aplicabilidad de negocio
4. Distribución balanceada (19%, 60%, 21%)

**Nota:** Silhouette 0.19 es esperado en datos reales por solapamiento natural

### 3.2 Convergencia
```python
max_iterations = 100
tolerance = 1e-4
random_state = 42
```

**Justificación:**
- max_iterations: Suficiente (converge en 8-15 iteraciones)
- tolerance: Detecta convergencia cuando cambio < 0.0001
- random_state: Reproducibilidad

### 3.3 Inicialización

**K-Means++**: Distribuye centroides iniciales lejos entre sí

**Ventajas:**
- Reduce mínimos locales pobres
- Converge más rápido

### 3.4 TF-IDF

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| max_features | 100 | Balance información/ruido |
| min_df | 2 | Elimina typos |
| max_df | 0.8 | Elimina stopwords |
| ngram_range | (1,2) | Captura frases |

---

## 4. Documentación del Flujo del Proyecto y Herramientas

### 4.1 Tecnologías

#### Backend

| Categoría | Tecnología |
|-----------|------------|
| Lenguaje | Python 3.11+ |
| Framework | FastAPI |
| Servidor | Uvicorn |
| Datos | Pandas, NumPy |
| ML | scikit-learn |
| Visualización | Matplotlib, ReportLab, Pillow |

#### Frontend

| Categoría | Tecnología |
|-----------|------------|
| Framework | Next.js 14 |
| Lenguaje | TypeScript |
| UI | React, Tailwind CSS |
| Gráficos | Recharts |
| HTTP Client | Axios |

### 4.2 Flujo de Datos
```
1. Carga CSV → Validación columnas
2. Preprocesamiento → Limpieza + Normalización
3. Clustering Clientes → K-Means numérico
4. Clustering Texto → TF-IDF + K-Means
5. Evaluación → Métricas (Silhouette, Davies-Bouldin)
6. Perfiles → Descripciones en lenguaje natural
7. Visualización → PCA 2D
8. Exportación → CSV + PDF + PNG
```

### 4.3 Arquitectura
```
Frontend (Next.js) ←→ Backend (FastAPI) ←→ Módulos ML ←→ Exportación
```

---

## 5. Conclusiones y Lecciones Aprendidas

### 5.1 Logros

- K-Means desde cero funcional
- Pipeline robusto de preprocesamiento
- Enfoque dual (numérico + texto)
- Perfiles automáticos en lenguaje natural
- Visualización PCA profesional
- Exportación PDF con gráficos
- Interfaz web interactiva

### 5.2 Desafíos y Soluciones

**Desafío 1: Datos sucios**
- Problema: Valores "Error", "abc", NaN dispersos
- Solución: Estrategia híbrida (eliminar críticos, rellenar opcionales)
- Lección: Preprocesamiento robusto es tan importante como el modelo

**Desafío 2: Métricas "pobres" pero realistas**
- Problema: Silhouette 0.19 (calificación "Pobre")
- Análisis: Segmentación real siempre tiene solapamiento
- Lección: No perseguir métricas perfectas en datos reales

**Desafío 3: Balance variables**
- Problema: 6 numéricas vs 5 categóricas (one-hot)
- Solución: Normalización asegura contribución equitativa
- Lección: StandardScaler crítico con one-hot encoding

### 5.3 Lecciones Técnicas

1. **Implementar desde cero no significa reinventar todo**: K-Means core implementado, utilidades estándar reutilizadas
2. **Calidad de datos > Sofisticación del modelo**: 50% del trabajo fue preprocesamiento
3. **Dominio del negocio > Métricas puras**: Clusters interpretables aunque métricas sean moderadas

### 5.4 Mejoras Futuras

**Corto plazo:**
- Método del codo automático
- Dashboard interactivo mejorado
- Análisis de sentimiento explícito

**Largo plazo:**
- Predicción para nuevos clientes
- Modelos ensemble
- Integración con CRM

---

## 6. API Endpoints

**Base URL:** `http://localhost:8000`

### 6.1 GET / - Información General
```json
{
  "message": "Bienvenido a InsightCluster API",
  "version": "1.0.0",
  "status": "running"
}
```

### 6.2 POST /upload - Subir Archivo

**Request:** multipart/form-data

**Columnas requeridas:**
- cliente_id
- frecuencia_compra
- monto_total_gastado
- canal_principal

**Response:**
```json
{
  "status": "success",
  "file_id": "20251227_143022",
  "rows": 500
}
```

### 6.3 GET /info/{file_id} - Info Archivo

Retorna metadatos del archivo cargado

### 6.4 POST /train - Entrenar Modelo

**Request:**
```json
{
  "file_id": "20251227_143022",
  "n_clusters": 3,
  "max_iterations": 100,
  "random_state": 42
}
```

**Response:**
```json
{
  "status": "success",
  "summary": {
    "n_clusters": 3,
    "overall_quality": "Bueno",
    "recommendation": "Clusters bien definidos"
  },
  "metrics": {
    "silhouette_score": 0.6234,
    "davies_bouldin_score": 0.7821
  },
  "clusters": [...],
  "reviews": {...}
}
```

### 6.5 GET /results/{file_id} - Obtener Resultados

Retorna resultados completos del modelo entrenado

### 6.6 GET /download/{file_id} - Descargar CSV

Descarga CSV con columna 'cluster' agregada

### 6.7 GET /metrics/info - Info Métricas

Retorna rangos e interpretación de métricas

### 6.8 GET /visualization/{file_id} - Generar PCA

Genera visualización 2D mediante PCA

**Response:**
```json
{
  "image_path": "data/visualizations/{file_id}_pcaGrafica.png",
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

### 6.9 GET /export/pdf/{file_id} - Exportar PDF

Genera reporte PDF profesional con:
1. Portada con resumen ejecutivo
2. Visualización PCA
3. Métricas y segmentos detallados

### 6.10 GET /health - Estado API

Verifica estado de la API

---

## 7. Estructura de Archivos

### 7.1 Árbol del Proyecto Completo
```
Proyecto 2/
├── Backend/
│   ├── app.py                          # FastAPI application principal
│   ├── requirements.txt                # Dependencias Python
│   ├── README.md                       # Documentación técnica
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── kmeans.py                  # K-Means desde cero
│   │   ├── preprocessing.py           # Pipeline preprocesamiento
│   │   ├── text_clustering.py         # Clustering texto (TF-IDF)
│   │   ├── profiling.py               # Generación perfiles clusters
│   │   └── generate_test_data.py      # Generador datos sintéticos
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   └── metrics.py                 # Métricas evaluación
│   │
│   └── data/
│       ├── uploads/                   # CSVs cargados
│       ├── exports/                   # CSVs con clusters
│       ├── visualizations/            # Gráficas PCA (PNG)
│       └── reports/                   # Reportes PDF
│
└── Frontend/
    ├── public/                        # Archivos estáticos
    ├── src/
    │   └── app/
    │       ├── components/            # Componentes React
    │       │   ├── header/           # Header de navegación
    │       │   └── sidebar/          # Sidebar de navegación
    │       │
    │       ├── pages/                # Páginas de la aplicación
    │       │   ├── carga-masiva/            # Carga de archivos CSV
    │       │   ├── configuracion-modelo/    # Configuración K-Means
    │       │   ├── evaluacion-modelos/      # Visualización de métricas
    │       │   ├── interpretacion-segmentos/ # Análisis de clusters
    │       │   └── exportacion-reportes/    # Descarga de resultados
    │       │
    │       └── services/             # Servicios API
    │
    ├── package.json                   # Dependencias Node.js
    └── next.config.js                # Configuración Next.js
```

### 7.2 Descripción de Archivos Backend

#### app.py

**Responsabilidades:**
- 10 endpoints FastAPI
- Orquestación de módulos
- Generación visualizaciones PCA
- Exportación PDF
- Almacenamiento temporal

**Imports:**
```python
from models.preprocessing import DataPreprocessor
from models.kmeans import KMeans
from models.text_clustering import TextClusterer
from utils.metrics import ClusteringMetrics
from models.profiling import ClusterProfiler
```

---

#### models/kmeans.py

**Clase:** `KMeans`

**Métodos:**
```python
def fit(X)                     # Entrenar modelo
def predict(X)                 # Predecir clusters
def _calculate_distances(X)    # Distancias euclidianas
def _calculate_inertia(X)      # WCSS
```

**Algoritmo:**
1. Inicialización aleatoria centroides
2. Asignación a centroide más cercano
3. Recálculo de centroides
4. Repetir hasta convergencia

**Complejidad:** O(n · k · i)

---

#### models/preprocessing.py

**Clase:** `DataPreprocessor`

**Métodos:**
```python
def load_and_clean(file_path)           # Pipeline completo
def clean_review_text(text)             # Limpieza texto
def calculate_cluster_stats(df, labels) # Estadísticas clusters
```

**Tareas:**
1. Lectura CSV
2. Limpieza texto
3. Conversión valores no numéricos
4. Eliminación/relleno NaN
5. Detección outliers
6. One-hot encoding
7. Normalización

---

#### models/text_clustering.py

**Clase:** `TextClusterer`

**Métodos:**
```python
def fit_predict(texts)          # Entrenar y predecir
def get_top_terms(n_terms=5)    # Top keywords
def get_cluster_stats(labels)   # Distribución clusters
```

**Pipeline:**
1. Limpieza texto
2. Vectorización TF-IDF (954 × 100)
3. K-Means scikit-learn
4. Extracción keywords

---

#### models/profiling.py

**Clase:** `ClusterProfiler`

**Métodos:**
```python
def generate_profiles()                     # Perfiles completos
def _build_description(cluster)            # Descripción texto
def _qualitative_level(value, mean)        # "alto"/"medio"/"bajo"
def _get_review_topics(cluster_id)         # Keywords reseñas
```

**Salida:**
```python
{
  "cluster_id": 0,
  "description": "Este segmento agrupa clientes con frecuencia alta...",
  "size": 150,
  "canal_principal": "web"
}
```

---

#### models/generate_test_data.py

**Función:** `generate_test_data(n_clients=200, n_reviews=500)`

**Genera:**
- Clientes con datos aleatorios
- 10 plantillas de reseñas
- CSV en data/uploads/test_data.csv

**Uso:**
```bash
python models/generate_test_data.py
```

---

#### utils/metrics.py

**Clase:** `ClusteringMetrics`

**Métodos:**
```python
@staticmethod
def calculate_all_metrics(X, labels)      # 3 métricas
def _interpret_metrics(metrics)           # Interpretación
def get_metric_ranges()                   # Info métricas
```

**Métricas:**

1. **Silhouette Score** ([-1, 1], mayor=mejor)
2. **Calinski-Harabasz** ([0, ∞), mayor=mejor)
3. **Davies-Bouldin** ([0, ∞), menor=mejor)

---

### 7.3 Descripción de Páginas Frontend

#### pages/carga-masiva/

**Funcionalidad:**
- Upload de archivos CSV/Excel
- Validación de formato
- Vista previa de datos
- Confirmación de carga

**Comunicación con Backend:**
- POST /upload

---

#### pages/configuracion-modelo/

**Funcionalidad:**
- Selección de número de clusters (K)
- Configuración de max_iterations
- Configuración de random_state
- Inicio de entrenamiento

**Comunicación con Backend:**
- POST /train

---

#### pages/evaluacion-modelos/

**Funcionalidad:**
- Visualización de métricas
- Gráficos de calidad
- Interpretación automática
- Recomendaciones

**Comunicación con Backend:**
- GET /results/{file_id}
- GET /metrics/info

---

#### pages/interpretacion-segmentos/

**Funcionalidad:**
- Visualización de clusters
- Descripciones en lenguaje natural
- Estadísticas por segmento
- Keywords de reseñas
- Gráfico PCA

**Comunicación con Backend:**
- GET /results/{file_id}
- GET /visualization/{file_id}

---

#### pages/exportacion-reportes/

**Funcionalidad:**
- Descarga CSV con clusters
- Generación y descarga de PDF
- Descarga de visualización PCA
- Resumen de resultados

**Comunicación con Backend:**
- GET /download/{file_id}
- GET /export/pdf/{file_id}

---

### 7.4 Flujo de Conexión Backend-Frontend
```
Frontend                    Backend
   │
   ├─→ Carga CSV          → POST /upload
   │                         └─→ Almacena en data/uploads/
   │
   ├─→ Configurar K       → POST /train
   │                         ├─→ DataPreprocessor
   │                         ├─→ KMeans.fit()
   │                         ├─→ TextClusterer
   │                         ├─→ ClusteringMetrics
   │                         └─→ ClusterProfiler
   │
   ├─→ Ver Resultados     → GET /results/{file_id}
   │                         └─→ Retorna JSON con clusters
   │
   ├─→ Ver Gráfico PCA    → GET /visualization/{file_id}
   │                         ├─→ PCA(n_components=2)
   │                         ├─→ matplotlib scatter
   │                         └─→ Retorna ruta PNG
   │
   └─→ Descargar          → GET /download/{file_id}
                          → GET /export/pdf/{file_id}
                            └─→ Retorna archivos
```

---

## 8. Instalación y Uso

### 8.1 Requisitos

**Backend:**
- Python 3.11+
- pip

**Frontend:**
- Node.js 18+
- npm o yarn

**General:**
- Git

### 8.2 Instalación Backend
```bash
# Clonar repositorio
git clone https://github.com/Carbonell-Castillo/OLC2_2SEVD25_ML_-30.git
cd "Proyecto 2/Backend"

# Crear entorno virtual
python -m venv venv

# Activar entorno
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Instalar dependencias
pip install -r requirements.txt
```

### 8.3 Instalación Frontend
```bash
# Ir a carpeta frontend
cd "Proyecto 2/Frontend"

# Instalar dependencias
npm install
# o
yarn install
```

### 8.4 Ejecución

**Backend:**
```bash
cd "Proyecto 2/Backend"
uvicorn app:app --reload

# Servidor en: http://localhost:8000
# Documentación: http://localhost:8000/docs
```

**Frontend:**
```bash
cd "Proyecto 2/Frontend"
npm run dev
# o
yarn dev

# Aplicación en: http://localhost:3000
```

### 8.5 Generar Datos de Prueba
```bash
cd "Proyecto 2/Backend"
python models/generate_test_data.py
```

### 8.6 Flujo de Uso Completo
```
1. Iniciar Backend (puerto 8000)
2. Iniciar Frontend (puerto 3000)
3. Abrir http://localhost:3000
4. Cargar CSV en "Carga Masiva"
5. Configurar modelo en "Configuración Modelo"
6. Ver métricas en "Evaluación Modelos"
7. Analizar segmentos en "Interpretación Segmentos"
8. Descargar resultados en "Exportación Reportes"
```

---

## Resultados Obtenidos

### Clustering de Clientes (K=3)

| Cluster | Tamaño | % | Canal | Perfil |
|---------|--------|---|-------|--------|
| 0 | 181 | 19% | Web | VIP Digital |
| 1 | 573 | 60% | Tienda | Mayoría Regular |
| 2 | 200 | 21% | Call Center | Premium |

### Clustering de Reseñas (K=3)

| Cluster | % | Keywords | Sentimiento |
|---------|---|----------|-------------|
| 0 | 44% | cumple, bien, interfaz | Positivo |
| 1 | 40% | aunque, atención, poco | Neutro |
| 2 | 16% | uso, mejorar, rendimiento | Constructivo |

### Métricas

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| Silhouette | 0.1863 | Aceptable |
| Davies-Bouldin | 1.9304 | Solapamiento presente |

---

## Dependencias

### Backend (requirements.txt)
```
fastapi==0.104.1
uvicorn==0.24.0
pandas==2.1.3
numpy==1.26.2
scikit-learn==1.3.2
matplotlib==3.8.2
reportlab==4.0.7
Pillow==12.0.0
python-multipart==0.0.6
```

### Frontend (package.json)
```json
{
  "dependencies": {
    "next": "14.x",
    "react": "18.x",
    "react-dom": "18.x",
    "typescript": "5.x",
    "tailwindcss": "3.x",
    "recharts": "2.x",
    "axios": "1.x"
  }
}
```

---

## Autores

Madeline Fabiola Prado Reyes 202100039
Josue Daniel Chavez Portillo 202100033
Bruce Carbonell Castillo Cifuentes 202203069
Universidad de San Carlos de Guatemala  
Organización de Lenguajes y Compiladores 2  
Diciembre 2025

---

## Repositorio

**GitHub:** https://github.com/Carbonell-Castillo/OLC2_2SEVD25_ML_-30  
**Rama:** `develop`  
**Carpeta Backend:** `Proyecto 2/Backend/`  
**Carpeta Frontend:** `Proyecto 2/Frontend/`