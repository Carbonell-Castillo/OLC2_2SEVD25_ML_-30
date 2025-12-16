import os
import joblib
import numpy as np

from typing import Dict
from src.ml.training import SAVED_MODELS_DIR, MODEL_FILENAME

# Las columnas que el modelo espera, EN EL MISMO ORDEN
FEATURE_COLUMNS = [
    "asistencia_clases",
    "tareas_entregadas",
    "participacion_clase",
    "horas_estudio",
    "promedio_evaluaciones",
    "cursos_reprobados",
    "actividades_extracurriculares",
    "reportes_disciplinarios",
    "promedio_actual",
]


def load_trained_model():
    """
    Carga el modelo entrenado desde disco.
    Lanza ValueError si el modelo no existe.
    """
    model_path = os.path.join(SAVED_MODELS_DIR, MODEL_FILENAME)

    if not os.path.exists(model_path):
        raise ValueError(
            f"No se encontró el modelo entrenado en '{model_path}'. "
            "Primero debes llamar a /api/train."
        )

    model = joblib.load(model_path)
    return model, model_path


def predict_risk(input_data: Dict):
    """
    Recibe un diccionario con los datos de UN estudiante y
    devuelve la predicción de riesgo.

    input_data debe tener, al menos, estas llaves numéricas:
        - asistencia_clases
        - tareas_entregadas
        - participacion_clase
        - horas_estudio
        - promedio_evaluaciones
        - cursos_reprobados
        - actividades_extracurriculares
        - reportes_disciplinarios
        - promedio_actual
    """

    # 1) Verificar que el modelo exista y cargarlo
    model, model_path = load_trained_model()

    # 2) Verificar que vengan todas las columnas necesarias
    missing = [col for col in FEATURE_COLUMNS if col not in input_data]
    if missing:
        raise ValueError(
            f"Faltan los siguientes campos en el JSON de entrada: {', '.join(missing)}"
        )

    # 3) Construir el vector de entrada en el MISMO ORDEN
    try:
        values = [float(input_data[col]) for col in FEATURE_COLUMNS]
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"Todos los campos deben ser numéricos. Detalle: {str(e)}"
        )

    X = np.array([values])  # shape (1, n_features)

    # 4) Hacer la predicción
    pred = model.predict(X)[0]

    # Probabilidad de clase "1" (riesgo), si el modelo lo soporta
    prob_risk = None
    if hasattr(model, "predict_proba"):
        prob_risk = float(model.predict_proba(X)[0][1])

    # 5) Armar respuesta
    return {
        "input_used": {col: float(input_data[col]) for col in FEATURE_COLUMNS},
        "prediction": int(pred),
        "prediction_label": "riesgo" if int(pred) == 1 else "no_riesgo",
        "probability_riesgo": prob_risk,
        "model_path": model_path,
    }