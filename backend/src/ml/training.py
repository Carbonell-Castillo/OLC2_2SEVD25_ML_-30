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

from src.config import FEATURE_COLUMNS, TARGET_COLUMN

# Carpeta donde se guardará el modelo
SAVED_MODELS_DIR = "saved_models"
MODEL_FILENAME = "studentguard_model.pkl"


def train_model(df: pd.DataFrame):
    
    return train_model_with_params(df, hyperparams=None)


def train_model_with_params(df: pd.DataFrame, hyperparams: dict = None):

    # Valores por defecto de hiperparámetros
    if hyperparams is None:
        hyperparams = {}
    
    max_iter = hyperparams.get('max_iter', 1000)
    C = hyperparams.get('C', 0.5)
    solver = hyperparams.get('solver', 'lbfgs')

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"La columna '{TARGET_COLUMN}' no está presente en los datos limpios.")

    # Verificar que existan todas las columnas necesarias
    missing_cols = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Faltan las siguientes columnas: {', '.join(missing_cols)}")

    # Extraer  las columnas necesairas
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    print(f"\n Entrenando con {len(FEATURE_COLUMNS)} features:")
    for i, col in enumerate(FEATURE_COLUMNS, 1):
        print(f"   {i}. {col}")
    
    print(f"\n Total de muestras: {len(X)}")
    print(f"   - Riesgo (1): {int(y.sum())} ({100*y.mean():.1f}%)")
    print(f"   - No Riesgo (0): {int((1-y).sum())} ({100*(1-y.mean()):.1f}%)")

    print(f"\n Hiperparámetros del modelo:")
    print(f"   - max_iter: {max_iter}")
    print(f"   - C (regularización): {C}")
    print(f"   - solver: {solver}")

    # Convertir a numpy
    X = X.values
    y = y.values

    # Separar train / test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y 
    )

    # Definir modelo con hiperparámetros personalizados
    model = LogisticRegression(
        max_iter=max_iter,
        C=C,
        solver=solver,
        random_state=42
    )

    # Entrenar
    print("\n Entrenando modelo...")
    model.fit(X_train, y_train)
    print("    Entrenamiento completado")

    # Predecir en test
    y_pred = model.predict(X_test)

    # Calcular métricas
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "hyperparams_used": {
            "max_iter": max_iter,
            "C": C,
            "solver": solver
        }
    }

    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

    model_path = os.path.join(SAVED_MODELS_DIR, MODEL_FILENAME)
    joblib.dump(model, model_path)

    metrics["model_path"] = model_path

    print(f"\n Modelo guardado en: {model_path}")
    print(f"\n Métricas del modelo:")
    print(f"   • Accuracy:  {metrics['accuracy']:.3f}")
    print(f"   • Precision: {metrics['precision']:.3f}")
    print(f"   • Recall:    {metrics['recall']:.3f}")
    print(f"   • F1-Score:  {metrics['f1_score']:.3f}")

    return metrics