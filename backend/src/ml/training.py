import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Carpeta donde se guardará el modelo
SAVED_MODELS_DIR = "saved_models"
MODEL_FILENAME = "studentguard_model.pkl"


def train_model(df: pd.DataFrame):
    """
    Entrena un modelo de clasificación de riesgo (0 = no riesgo, 1 = riesgo)
    usando los datos LIMPIOS que ya generaste con DataCleaner.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con los datos LIMPIOS. Debe incluir la columna 'riesgo'
        como variable objetivo (0/1).

    Returns
    -------
    dict
        Diccionario con las métricas del modelo y la ruta donde se guardó:
        {
            "accuracy": ...,
            "precision": ...,
            "recall": ...,
            "f1_score": ...,
            "n_train": ...,
            "n_test": ...,
            "model_path": "saved_models/studentguard_model.pkl"
        }
    """

    # Asegurar que exista la columna 'riesgo'
    if "riesgo" not in df.columns:
        raise ValueError("La columna 'riesgo' no está presente en los datos limpios.")

    # 1) Quedarnos SOLO con columnas numéricas
    #    (ignoramos carnet, first_name, last_name, gender, etc.)
    numeric_df = df.select_dtypes(include=[np.number]).copy()

    if "riesgo" not in numeric_df.columns:
        raise ValueError("La columna 'riesgo' no es numérica después de la limpieza.")

    # 2) Separar features (X) y etiqueta (y)
    X = numeric_df.drop(columns=["riesgo"])
    y = numeric_df["riesgo"]

    # Convertir a numpy (por si acaso)
    X = X.values
    y = y.values

    # 3) Separar train / test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y  # mantiene proporción de clases
    )

    # 4) Definir modelo (Logistic Regression como base)
    model = LogisticRegression(max_iter=1000)

    # 5) Entrenar
    model.fit(X_train, y_train)

    # 6) Predecir en test
    y_pred = model.predict(X_test)

    # 7) Calcular métricas
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }

    # 8) Asegurar que exista la carpeta de modelos
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

    # 9) Guardar modelo entrenado
    model_path = os.path.join(SAVED_MODELS_DIR, MODEL_FILENAME)
    joblib.dump(model, model_path)

    metrics["model_path"] = model_path

    return metrics