"""
Configuración de columnas para StudentGuard
Este archivo define qué columnas usar para entrenar el modelo
"""

# Columnas que SÍ debe usar el modelo (en este orden exacto)
FEATURE_COLUMNS = [
    "promedio_actual",
    "asistencia_clases",
    "tareas_entregadas",
    "participacion_clase",
    "horas_estudio",
    "promedio_evaluaciones",
    "cursos_reprobados",
    "actividades_extracurriculares",
    "reportes_disciplinarios",
]

# La columna que queremos predecir
TARGET_COLUMN = "riesgo"

# Todas las columnas que debe tener el CSV
REQUIRED_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]