

# StudentGuard - Sistema de Predicción de Riesgo Académico

## Descripción del Proyecto

StudentGuard es un sistema de Machine Learning diseñado para predecir el riesgo académico de estudiantes universitarios. Utilizando Regresión Logística, el sistema analiza 9 variables académicas y de comportamiento para clasificar estudiantes en dos categorías: **"riesgo"** o **"no riesgo"**.

**Desarrollado por:** 
202100039 Fabiola
220203069 Bruce
**Curso:** Compiladores 2
**Universidad:** Universidad de San Carlos de Guatemala (USAC)
**Auxiliar:** Katherine Gomez

-----

## Objetivo

Proporcionar una herramienta automatizada que permita a instituciones educativas:

  * Identificar estudiantes en riesgo académico de forma temprana
  * Tomar decisiones informadas sobre intervenciones educativas
  * Optimizar recursos de apoyo estudiantil

-----

## Arquitectura del Sistema

### **Backend (Flask API)**

```
backend/
├── src/
│   ├── app/
│   │   └── app.py            # Servidor Flask con endpoints
│   ├── data/
│   │   ├── data_loader.py    # Carga y validación de CSV
│   │   └── data_cleaner.py   # Limpieza de datos
│   ├── ml/
│   │   ├── training.py       # Entrenamiento del modelo
│   │   └── prediction.py     # Predicciones
│   └── config.py             # Configuración de variables
├── uploads/                   # Archivos CSV temporales
├── saved_models/              # Modelos entrenados (.pkl)
└── requirements.txt          # Dependencias Python
```

## *Stack Tecnológico Completo*

#### *Lenguaje de Programación*

*Python 3.8+*
- *Justificación:* 
  - Lenguaje estándar en Data Science y Machine Learning
  - Amplio ecosistema de librerías especializadas
  - Sintaxis clara y mantenible
  - Comunidad activa y documentación extensa


#### *Framework Web*

*Flask 3.0.0*
- *Propósito:* Backend API REST
- *Justificación:*
  - Ligero y flexible (microframework)
  - Fácil de configurar y desplegar
  - Ideal para APIs de Machine Learning
  - No requiere infraestructura compleja
- *Características utilizadas:*
  - Routing de endpoints
  - Manejo de JSON
  - Upload de archivos multipart/form-data
  - Manejo de errores personalizado

*Flask-CORS 4.0.0*
- *Propósito:* Habilitar Cross-Origin Resource Sharing
- *Justificación:*
  - Permite que el frontend (en otro puerto/dominio) consuma la API
  - Esencial para arquitectura frontend-backend separada
  - Configuración simple

#### *Procesamiento de Datos*

*Pandas 2.1.4*
- *Propósito:* Manipulación y análisis de datos
- *Justificación:*
  - Estándar de la industria para datos tabulares
  - Optimizado para datasets medianos (1K-100K filas)
  - Funciones integradas para limpieza de datos
- *Funciones clave utilizadas:*
  - read_csv(): Carga de archivos CSV
  - fillna(): Imputación de valores nulos
  - drop_duplicates(): Eliminación de duplicados
  - to_numeric(): Conversión de tipos
  - describe(): Estadísticas descriptivas

*NumPy 1.26.2*
- *Propósito:* Operaciones numéricas y arrays
- *Justificación:*
  - Base de Pandas y scikit-learn
  - Operaciones vectorizadas eficientes
  - Funciones matemáticas optimizadas
- *Funciones clave utilizadas:*
  - clip(): Ajuste de valores a rangos
  - median(): Cálculo de medianas
  - Arrays numéricos para el modelo

---

### **Frontend (Angular)**

El frontend fue desarrollado usando **Angular** y se divide en las siguientes secciones, permitiendo al usuario interactuar con las funcionalidades del backend:

```
OLC2_2SEVD25_ML_-33
├─ Frontend
│  ├─ .angular
│  │  └─ cache
│  │     └─ 21.0.3
│  │        └─ Frontend
│  ├─ .editorconfig
│  ├─ angular.json
│  ├─ package-lock.json
│  ├─ package.json
│  ├─ public
│  │  └─ favicon.ico
│  ├─ README.md
│  ├─ src
│  │  ├─ app
│  │  │  ├─ app.config.server.ts
│  │  │  ├─ app.config.ts
│  │  │  ├─ app.css
│  │  │  ├─ app.html
│  │  │  ├─ app.routes.server.ts
│  │  │  ├─ app.routes.ts
│  │  │  ├─ app.spec.ts
│  │  │  ├─ app.ts
│  │  │  ├─ components
│  │  │  │  ├─ dashboard-layout.css
│  │  │  │  ├─ dashboard-layout.html
│  │  │  │  ├─ dashboard-layout.spec.ts
│  │  │  │  ├─ dashboard-layout.ts
│  │  │  │  ├─ header
│  │  │  │  │  ├─ header.css
│  │  │  │  │  ├─ header.html
│  │  │  │  │  ├─ header.spec.ts
│  │  │  │  │  └─ header.ts
│  │  │  │  └─ sidebar
│  │  │  │     ├─ sidebar.css
│  │  │  │     ├─ sidebar.html
│  │  │  │     ├─ sidebar.spec.ts
│  │  │  │     └─ sidebar.ts
│  │  │  ├─ pages
│  │  │  │  ├─ ajuste-parametros
│  │  │  │  │  ├─ ajuste-parametros.css
│  │  │  │  │  ├─ ajuste-parametros.html
│  │  │  │  │  ├─ ajuste-parametros.spec.ts
│  │  │  │  │  └─ ajuste-parametros.ts
│  │  │  │  ├─ carga-masiva
│  │  │  │  │  ├─ carga-masiva.css
│  │  │  │  │  ├─ carga-masiva.html
│  │  │  │  │  ├─ carga-masiva.spec.ts
│  │  │  │  │  └─ carga-masiva.ts
│  │  │  │  ├─ evaluacion-modelos
│  │  │  │  │  ├─ evaluacion-modelos.css
│  │  │  │  │  ├─ evaluacion-modelos.html
│  │  │  │  │  ├─ evaluacion-modelos.spec.ts
│  │  │  │  │  └─ evaluacion-modelos.ts
│  │  │  │  └─ prediccion-riesgo
│  │  │  │     ├─ prediccion-riesgo.css
│  │  │  │     ├─ prediccion-riesgo.html
│  │  │  │     ├─ prediccion-riesgo.spec.ts
│  │  │  │     └─ prediccion-riesgo.ts
│  │  │  └─ services
│  │  │     └─ data.service.ts
│  │  ├─ index.html
│  │  ├─ main.server.ts
│  │  ├─ main.ts
│  │  ├─ server.ts
│  │  └─ styles.css
│  ├─ tsconfig.app.json
│  ├─ tsconfig.json
│  └─ tsconfig.spec.json
```

  * **Carga Masiva**
    Se realiza la carga de datos de manera local
    ![alt text](image.png)
  * **Ajuste de Parámetros**
    Se ajustan los hiperparámetros para el cálculo de la predicción
![alt text](image-1.png)
  * **Evaluación de Modelos**
    Se evalúa el modelo respectivamente
    ![alt text](image-2.png)
  * **Predicción**
    Realiza la predicción dependiendo de los datos de entrada
    ![alt text](image-3.png)
### **Modelo de Machine Learning**

  * **Algoritmo:** Regresión Logística (Logistic Regression)
  * **Biblioteca:** scikit-learn
  * **Features:** 9 variables numéricas
  * **Target:** Variable binaria (0 = no riesgo, 1 = riesgo)

-----

## Flujo Operacional

El sistema sigue un flujo lineal y controlado que asegura la calidad y el entrenamiento adecuado del modelo antes de generar predicciones.

1.  **Carga Masiva (Upload):** El usuario sube el archivo CSV de estudiantes (`/api/upload`).
2.  **Limpieza de Datos (Clean):** Se aplica el proceso de preprocesamiento (eliminación de duplicados, manejo de nulos y ajuste de rangos) (`/api/clean`).
3.  **Entrenamiento (Train):** Se entrena el modelo de Regresión Logística, ya sea con parámetros por defecto o personalizados (`/api/train` o `/api/train_with_params`).
4.  **Evaluación (Evaluate):** Se muestran las métricas de rendimiento (Accuracy, F1-Score) para validar la calidad del modelo.
5.  **Predicción (Predict):** Una vez que el modelo está entrenado y validado, se utiliza para predecir el riesgo de estudiantes nuevos o individuales (`/api/predict`).

![alt text](diagrama.png)
-----

## Justificación del Modelo y Parámetros

### Modelo Seleccionado: Regresión Logística

La Regresión Logística (*Logistic Regression*) se seleccionó por las siguientes razones clave:

1.  **Problema de Clasificación Binaria:** El objetivo del sistema es clasificar a los estudiantes en solo dos categorías mutuamente excluyentes: **Riesgo (1)** o **No Riesgo (0)**. La Regresión Logística es el algoritmo lineal fundamental para este tipo de problemas de clasificación binaria.
2.  **Interpretabilidad:** A diferencia de modelos más complejos (como Redes Neuronales o *Random Forest*), la Regresión Logística es altamente **interpretable**. Esto significa que podemos entender la influencia de cada variable predictora (ej. `cursos_reprobados`, `asistencia_clases`) en la probabilidad de riesgo, lo cual es vital para el personal educativo que necesita tomar decisiones informadas.
3.  **Eficiencia Computacional:** Es un modelo rápido de entrenar y predecir, lo que lo hace ideal para un sistema que requiere procesar rápidamente grandes volúmenes de datos cargados masivamente.

$$P(Y=1 | X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \dots + \beta_n x_n)}}$$

### Justificación de Hiperparámetros

Los hiperparámetros por defecto se ajustaron para optimizar la convergencia y prevenir el *overfitting* de manera eficiente:

| Hiperparámetro | Valor | Justificación |
| :--- | :--- | :--- |
| **`max_iter`** | `1000` | Aumentado de su valor por defecto (`100` en scikit-learn). Un número mayor de iteraciones asegura que el algoritmo de optimización (`lbfgs`) tenga suficiente tiempo para **converger** y encontrar la mejor solución para los coeficientes, especialmente con *datasets* grandes. |
| **`C`** | `0.5` | Es el inverso del parámetro de regularización ($\frac{1}{\lambda}$). Un valor de `0.5` (que implica una regularización moderada) ayuda a **prevenir el *overfitting***, penalizando coeficientes grandes. Esto asegura que el modelo generalice bien a nuevos estudiantes en lugar de memorizar el *dataset* de entrenamiento. |
| **`solver`** | `lbfgs` | Es el algoritmo de optimización predeterminado y recomendado por `scikit-learn` para la mayoría de los casos de uso. Es eficiente y funciona bien con conjuntos de datos medianos como el usado en este proyecto. |
| **`random_state`** | `42` | Se utiliza para **fijar la semilla aleatoria**. Esto es crucial para la **reproducibilidad**; garantiza que la división de datos y el proceso de entrenamiento produzcan exactamente los mismos resultados en ejecuciones posteriores. |

-----
[11:48 p.m., 15/12/2025] Fabiola Fiusac:  Naturaleza del Problema*
- *Clasificación binaria:* El problema requiere predecir una de dos clases (riesgo/no riesgo)
- *Variables numéricas continuas:* Las 9 features son valores numéricos que mantienen relaciones lineales con la variable objetivo
- *Interpretabilidad requerida:* En un contexto educativo, es crucial poder explicar POR QUÉ un estudiante está en riesgo
##  Variables del Modelo
[11:49 p.m., 15/12/2025] Fabiola Fiusac: ### *Selección del Modelo*

*Algoritmo elegido:* Regresión Logística (Logistic Regression)

### *Justificación de la Selección*

*¿Por qué Logistic Regression y no otros modelos?*

Se evaluaron diferentes algoritmos de clasificación y se seleccionó Logistic Regression por las siguientes razones:
#### *1. Naturaleza del Problema*
- *Clasificación binaria:* El problema requiere predecir una de dos clases (riesgo/no riesgo)
- *Variables numéricas continuas:* Las 9 features son valores numéricos que mantienen relaciones lineales con la variable objetivo
- *Interpretabilidad requerida:* En un contexto educativo, es crucial poder explicar POR QUÉ un estudiante está en riesgo

## Variables del Modelo

### **Features (Variables Predictoras)**

| Variable | Tipo | Rango | Descripción |
| :--- | :--- | :--- | :--- |
| `promedio_actual` | float | 0-100 | Promedio de calificaciones actual del estudiante |
| `asistencia_clases` | float | 0-100 | Porcentaje de asistencia a clases |
| `tareas_entregadas` | float | 0-100 | Porcentaje de tareas entregadas |
| `participacion_clase` | float | 0-100 | Nivel de participación en clase (porcentaje) |
| `horas_estudio` | float | 0-24 | Horas de estudio diarias |
| `promedio_evaluaciones` | float | 0-100 | Promedio en evaluaciones (exámenes, quizzes) |
| `cursos_reprobados` | int | 0+ | Cantidad de cursos reprobados |
| `actividades_extracurriculares` | int | 0+ | Cantidad de actividades extracurriculares |
| `reportes_disciplinarios` | int | 0+ | Cantidad de reportes disciplinarios |

### **Target (Variable Objetivo)**

| Variable | Tipo | Valores | Descripción |
| :--- | :--- | :--- | :--- |
| `riesgo` | string/int | "riesgo" / "no riesgo" (convertido a 1/0) | Clasificación del estudiante |

-----

## Proceso de Limpieza de Datos

### **1. Eliminación de Duplicados**

**Decisión:** Eliminar filas completamente duplicadas

**Justificación:**

  * Filas 100% idénticas son errores de carga del sistema
  * NO aportan información nueva al modelo
  * Conservarlas inflaría artificialmente las estadísticas

**Ejemplo:**

```
Entrada: 210 filas
Duplicados encontrados: 3
Resultado: 207 filas únicas
```

-----

### **2. Tratamiento de Valores Nulos**

**Decisión:** Rellenar con la MEDIANA (no eliminar filas)

**Justificación:**

  * La mediana es más robusta ante valores extremos que la media
  * Evitamos perder datos valiosos (conservamos el 100% de las filas después de eliminar duplicados)
  * Es el método estándar en Machine Learning para datos numéricos

**Proceso:**

1.  Se calcula la mediana de cada columna numérica (excluyendo nulos)
2.  Los valores nulos se reemplazan con esta mediana
3.  Para variables categóricas, se usa la moda (valor más frecuente)

-----

### **3. Corrección de Valores Fuera de Rango**

**Decisión:** Ajustar al límite del rango válido (no eliminar)

**Justificación:**

  * Valores como `promedio_actual = 150` o `-10` son claramente errores de captura
  * Ajustar al límite (0 o 100) conserva el registro sin inventar datos arbitrarios
  * Es preferible a perder estudiantes completos del análisis

**Reglas de Ajuste:**

| Variable | Valor Incorrecto | Ajuste Aplicado |
| :--- | :--- | :--- |
| `promedio_actual` | \< 0 o \> 100 | → 0 o → 100 |
| `asistencia_clases` | \< 0 o \> 100 | → 0 o → 100 |
| `horas_estudio` | \< 0 o \> 24 | → 0 o → 24 |

-----

### **4. Tratamiento de Outliers (Valores Extremos)**

**Decisión:** CONSERVAR todos los outliers sin eliminar

**Justificación:**

En el contexto educativo, los valores extremos **SON VÁLIDOS Y NECESARIOS**:

  * Un estudiante con 100% en todo es un *outlier* válido (estudiante excelente).
  * Un estudiante con 0% asistencia es un *outlier* válido (abandono escolar).
  * Justamente los casos de **"riesgo alto"** suelen ser *outliers* que **QUEREMOS** identificar.

-----

### **5. Conversión de Tipos de Datos**

**Decisión:** Estandarizar formatos para el modelo

**Conversiones aplicadas:**

| Variable | Formato Original | Formato Final |
| :--- | :--- | :--- |
| `actividades_extracurriculares` | Lista de strings | Entero (cantidad) |
| `riesgo` | String | Entero binario |
| Todas las numéricas | String (si vienen así) | Float/Int |

-----

## Modelo de Machine Learning
*scikit-learn 1.3.2*
- *Propósito:* Algoritmos de Machine Learning
- *Justificación:*
  - Librería más madura y probada en producción
  - API consistente y bien documentada
  - Optimizaciones de rendimiento incluidas
  - Integración perfecta con Pandas/NumPy
- *Módulos utilizados:*

  *LogisticRegression*
  - Modelo de clasificación supervisada
  - Regularización L2 incorporada
  - Múltiples solvers disponibles
  
  *train_test_split*
  - División estratificada de datos
  - Asegura reproducibilidad (random_state)
  - Mantiene proporción de clases
  
  *Métricas (metrics)*
  - accuracy_score: Exactitud general
  - precision_score: Precisión (falsos positivos)
  - recall_score: Exhaustividad (falsos negativos)
  - f1_score: Balance Precision-Recall

### **Configuración del Modelo**

**Algoritmo:** Regresión Logística (Logistic Regression)

**Hiperparámetros por defecto:**

```python
{
    "max_iter": 1000,            # Iteraciones máximas
    "C": 0.5,                    # Parámetro de regularización
    "solver": "lbfgs",           # Algoritmo de optimización
    "random_state": 42           # Semilla para reproducibilidad
}
```

**División de Datos:**

  * **Entrenamiento:** 80% (train\_test\_split)
  * **Prueba:** 20%
  * **Estratificación:** Mantiene proporción de clases

-----

### **Métricas de Evaluación**

El modelo se evalúa con 4 métricas principales:

| Métrica | Descripción | Rango Esperado |
| :--- | :--- | :--- |
| **Accuracy** | Porcentaje de predicciones correctas | 85-95% |
| **Precision** | De los que predice "riesgo", cuántos realmente lo están | 84-94% |
| **Recall** | De los que están en riesgo, cuántos detecta | 82-92% |
| **F1-Score** | Balance entre Precision y Recall | 84-93% |

**Ejemplo de métricas obtenidas (CSV 1100 estudiantes):**

```
Accuracy:   89.5%
Precision: 84.6%
Recall:    85.7%
F1-Score:  85.2%
```

-----

### **Validación Anti-Overfitting**

Nuestro modelo muestra métricas entre 84-89%, lo cual es un **rango saludable**. Esto indica que **NO** tiene *overfitting* (memorización de datos) y generalizará bien con datos nuevos.

-----

## Instalación y Configuración

### **Requisitos Previos**

  * Python 3.8 o superior
  * pip (gestor de paquetes de Python)

### **Paso 1: Clonar el repositorio**

```bash
git clone <URL_DEL_REPOSITORIO>
cd backend
```

### **Paso 2: Crear entorno virtual (recomendado)**

```bash
python -m venv venv
```

**Activar entorno virtual:**

  * Windows: `venv\Scripts\activate`
  * Linux/Mac: `source venv/bin/activate`

### **Paso 3: Instalar dependencias**

```bash
pip install --break-system-packages -r requirements.txt
```

### **Paso 4: Crear carpetas necesarias**

```bash
mkdir uploads
mkdir saved_models
```

### **Paso 5: Ejecutar el servidor**

```bash
python -m src.app.app
```

El servidor estará disponible en: `http://localhost:5000`

-----

## API Endpoints

### **Base URL:** `http://localhost:5000`

(Se omiten detalles de los 10 endpoints por concisión en este resumen, pero se listan: Health Check, Cargar CSV, Limpiar Datos, Entrenar Modelo, Entrenar con Hiperparámetros, Predecir Riesgo, Obtener Información de Datos, Comparar Datos, Exportar Datos Limpios, Reiniciar Sistema).

-----

## Resultados y Validación

### **Dataset de Prueba**

  * Total de estudiantes: 1,100
  * Estudiantes en riesgo: 383 (34.8%)
  * División: 880 entrenamiento / 220 prueba

### **Métricas Obtenidas**

  * Accuracy: 89.5%
  * Precision: 84.6%
  * Recall: 85.7%
  * F1-Score: 85.2%

-----

## Conclusiones

1.  **Viabilidad y Precisión del Modelo:** El sistema StudentGuard, utilizando Regresión Logística, logró una alta tasa de acierto ($89.5\%$) y un buen balance entre *Precision* y *Recall* ($84.6\%$ y $85.7\%$ respectivamente). Esto valida el modelo como una herramienta efectiva para la **identificación temprana** de estudiantes en riesgo académico, cumpliendo el objetivo principal del proyecto.
2.  **Robustez en la Limpieza de Datos:** Las políticas de limpieza de datos (uso de la **mediana** para nulos y **ajuste a límites** para valores fuera de rango) demostraron ser robustas y cruciales. La decisión de **conservar los *outliers*** fue fundamental para asegurar que el modelo aprenda a identificar los casos de riesgo extremo, que son el objetivo principal del sistema.
3.  **Tecnología y Escalabilidad:** La arquitectura de microservicio con **Flask** para el backend, **scikit-learn** para ML y **Angular** para el frontend proporciona una base sólida. Esta separación permite la **escalabilidad** y la integración futura con sistemas universitarios (como se propone en las mejoras), ya que el frontend se puede reemplazar fácilmente y el backend puede ser consumido por múltiples plataformas.

-----

## Dependencias (requirements.txt)

```
Flask==3.0.0
flask-cors==4.0.0
pandas==2.1.4
numpy==1.26.2
scikit-learn==1.3.2
joblib==1.3.2
Werkzeug==3.0.1
```

-----

## Créditos

**Desarrollador:** Fabiola y Bruce
**Curso:** Compiladores 2
**Auxiliar:** Katherine Gomez
**Universidad:** Universidad de San Carlos de Guatemala (USAC)

