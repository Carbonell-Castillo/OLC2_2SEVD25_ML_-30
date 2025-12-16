import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class DataCleaner:
    """
    Esta clase limpia los datos que vienen del CSV para que estén listos para entrenar el modelo de machine learning.
    
    ¿Por qué necesitamos limpiar datos?
    - Los datos del mundo real vienen con errores (valores faltantes, errores de captura, etc.)
    - Si entrenamos con datos sucios, el modelo aprenderá mal
    - Es como estudiar con apuntes llenos de errores, vas a reprobar el examen
    
    ¿Qué problemas arregla?
    1. Valores faltantes (espacios en blanco)
    2. Valores fuera de rango (ej: promedio de 150 cuando el máximo es 100)
    3. Duplicados (el mismo estudiante registrado dos veces)
    4. Tipos de datos incorrectos (texto donde debería ir un número)
    5. NUEVO: Texto en columnas que deberían ser numéricas
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.cleaning_report = {}
    
    def clean_data(self, df):
        """
        Este es el proceso completo de limpieza.
        Ejecuta todos los pasos necesarios en orden.
        
        ORDEN CORREGIDO:
        1. Primero quitamos duplicados (no tiene sentido limpiar datos repetidos)
        2. Convertimos texto a números en columnas numéricas (NUEVO)
        3. Luego rellenamos valores faltantes
        4. Después corregimos valores fuera de rango
        
        Args:
            df: DataFrame original (posiblemente sucio)
        
        Returns:
            DataFrame limpio y listo para usar
        """
        print("🧹 Iniciando limpieza de datos...")
        df_clean = df.copy()
        
        # Paso 1: Eliminar filas duplicadas
        df_clean = self.remove_duplicates(df_clean)
        
        # Paso 2: Estandarizar tipos (convertir texto a números) - AHORA VA PRIMERO
        df_clean = self.standardize_data_types(df_clean)
        
        # Paso 3: Rellenar valores faltantes (ya después de convertir texto)
        df_clean = self.handle_missing_values(df_clean)
        
        # Paso 4: Corregir valores fuera de rango
        df_clean = self.fix_out_of_range_values(df_clean)
        
        print("✅ Limpieza completada")
        return df_clean
    
    def remove_duplicates(self, df):
        """
        Elimina estudiantes que aparecen más de una vez.
        """
        initial_rows = len(df)
        df_clean = df.drop_duplicates()
        removed = initial_rows - len(df_clean)
        
        self.cleaning_report['duplicates_removed'] = removed
        
        if removed > 0:
            print(f"  ➜ Eliminadas {removed} filas duplicadas")
        else:
            print(f"  ➜ No se encontraron duplicados")
        
        return df_clean
    
    def standardize_data_types(self, df):
        """
        Se asegura de que cada columna tenga el tipo de dato correcto.
        
        NUEVO: Ahora maneja el caso donde hay TEXTO en columnas numéricas
        Ejemplo: "Si", "No", "Alto", "Bajo" en lugar de números
        NUEVO 2: Maneja listas en formato string para actividades_extracurriculares
        
        ¿Por qué importa?
        - Si una columna de números se lee como texto, no podemos hacer cálculos
        - El modelo necesita números para aprender, no puede trabajar con texto
        - Previene errores raros más adelante
        """
        df_clean = df.copy()
        
        # === PASO ESPECIAL: Convertir listas de actividades a números ===
        if 'actividades_extracurriculares' in df_clean.columns:
            print("  ➜ Procesando actividades_extracurriculares (convirtiendo listas a cantidad)...")
            
            def count_activities(value):
                """Convierte listas de actividades a cantidad numérica"""
                if pd.isna(value):
                    return None
                
                # Si ya es un número, retornarlo
                if isinstance(value, (int, float)):
                    return value
                
                # Si es string, intentar parsearlo
                if isinstance(value, str):
                    # Casos: "[]", "['deportes']", "['deportes', 'club']"
                    # Limpiar y contar elementos
                    value_clean = value.strip()
                    
                    # Si está vacío o es "[]"
                    if value_clean == "[]" or value_clean == "":
                        return 0
                    
                    # Contar comas + 1 (aproximación rápida)
                    # "['deportes']" -> 0 comas = 1 actividad
                    # "['deportes', 'club']" -> 1 coma = 2 actividades
                    if "[" in value_clean and "]" in value_clean:
                        # Extraer contenido entre corchetes
                        content = value_clean.strip("[]")
                        if content.strip() == "":
                            return 0
                        # Contar elementos separados por comas
                        count = len([x for x in content.split(",") if x.strip()])
                        return count
                
                # Si no pudimos convertir, devolver None
                return None
            
            # Aplicar la conversión
            original_values = df_clean['actividades_extracurriculares'].copy()
            df_clean['actividades_extracurriculares'] = df_clean['actividades_extracurriculares'].apply(count_activities)
            
            converted = (original_values != df_clean['actividades_extracurriculares']).sum()
            print(f"    ✓ {converted} listas convertidas a cantidades numéricas")
        
        print("  ➜ Convirtiendo valores de texto a numéricos...")
        
        # Todas estas columnas DEBEN ser números
        numeric_columns = [
            'promedio_actual',
            'asistencia_clases',
            'tareas_entregadas',
            'participacion_clase',
            'horas_estudio',
            'promedio_evaluaciones',
            'cursos_reprobados',
            'actividades_extracurriculares',
            'reportes_disciplinarios',
            'riesgo'
        ]
        
        text_conversions = {}
        
        for col in numeric_columns:
            if col in df_clean.columns:
                # Guardar cuántos NaN había antes
                nulls_before = df_clean[col].isnull().sum()
                
                # Casos especiales antes de convertir a numérico
                if col == 'riesgo':
                    # Convertir texto a números: "riesgo" -> 1, "no riesgo" -> 0
                    df_clean[col] = df_clean[col].apply(
                        lambda x: 1 if str(x).lower().strip() == 'riesgo' 
                        else 0 if str(x).lower().strip() == 'no riesgo' 
                        else x
                    )
                
                # Convertir a numérico (texto se convierte a NaN)
                # errors='coerce' significa: si no se puede convertir, ponlo como NaN
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                
                # Calcular cuántos valores de texto se convirtieron a NaN
                nulls_after = df_clean[col].isnull().sum()
                text_converted = nulls_after - nulls_before
                
                if text_converted > 0:
                    text_conversions[col] = text_converted
                    print(f"    ⚠️  {col}: {text_converted} valores de texto convertidos a NaN (serán rellenados)")
        
        # Guardar reporte de conversiones
        self.cleaning_report['text_converted_to_numeric'] = text_conversions
        
        # La columna 'riesgo' además debe ser entero (0 o 1, no 0.5)
        if 'riesgo' in df_clean.columns:
            df_clean['riesgo'] = df_clean['riesgo'].fillna(0).astype(int)
        
        print("  ➜ Tipos de datos estandarizados correctamente")
        
        return df_clean
    
    def handle_missing_values(self, df):
        """
        Rellena los valores faltantes (NaN, None, vacíos)
        
        Estrategia que usamos:
        - Para números: usamos la MEDIANA (valor del medio)
          ¿Por qué mediana y no promedio? 
          Porque la mediana no se ve afectada por valores extremos
        """
        df_clean = df.copy()
        missing_before = df_clean.isnull().sum().sum()
        
        if missing_before == 0:
            print("  ➜ No hay valores faltantes que rellenar")
            self.cleaning_report['missing_values_handled'] = 0
            return df_clean
        
        # === PARTE 1: Rellenar columnas numéricas con la mediana ===
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        filled_count = 0
        for col in numeric_cols:
            if df_clean[col].isnull().any():
                missing_count = df_clean[col].isnull().sum()
                median_val = df_clean[col].median()
                df_clean[col].fillna(median_val, inplace=True)
                filled_count += missing_count
                print(f"  ➜ {col}: {missing_count} valores rellenados con mediana ({median_val:.2f})")
        
        # === PARTE 2: Rellenar columnas de texto con la moda ===
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if df_clean[col].isnull().any():
                missing_count = df_clean[col].isnull().sum()
                mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else "DESCONOCIDO"
                df_clean[col].fillna(mode_val, inplace=True)
                filled_count += missing_count
                print(f"  ➜ {col}: {missing_count} valores rellenados con moda ({mode_val})")
        
        self.cleaning_report['missing_values_handled'] = filled_count
        
        return df_clean
    
    def fix_out_of_range_values(self, df):
        """
        Corrige valores que no tienen sentido en el contexto educativo.
        
        Ejemplos de valores inválidos:
        - Promedio de 150 (el máximo es 100)
        - Asistencia de -10% (no puede ser negativa)
        - 200 horas de estudio semanal (solo hay 168 horas en una semana)
        """
        df_clean = df.copy()
        
        # Definimos los rangos lógicos para cada variable
        # (mínimo, máximo) - None significa sin límite
        ranges = {
            'promedio_actual': (0, 100),
            'asistencia_clases': (0, 100),
            'tareas_entregadas': (0, 100),
            'participacion_clase': (0, 100),
            'horas_estudio': (0, 168),
            'promedio_evaluaciones': (0, 100),
            'cursos_reprobados': (0, None),
            'actividades_extracurriculares': (0, None),
            'reportes_disciplinarios': (0, None),
            'riesgo': (0, 1)
        }
        
        total_adjusted = 0
        
        for col, (min_val, max_val) in ranges.items():
            if col in df_clean.columns:
                # Ajustar valores por debajo del mínimo
                if min_val is not None:
                    below_min = (df_clean[col] < min_val).sum()
                    if below_min > 0:
                        df_clean.loc[df_clean[col] < min_val, col] = min_val
                        total_adjusted += below_min
                        print(f"  ➜ {col}: {below_min} valores ajustados al mínimo ({min_val})")
                
                # Ajustar valores por encima del máximo
                if max_val is not None:
                    above_max = (df_clean[col] > max_val).sum()
                    if above_max > 0:
                        df_clean.loc[df_clean[col] > max_val, col] = max_val
                        total_adjusted += above_max
                        print(f"  ➜ {col}: {above_max} valores ajustados al máximo ({max_val})")
        
        if total_adjusted == 0:
            print("  ➜ No se encontraron valores fuera de rango")
        
        self.cleaning_report['values_adjusted'] = total_adjusted
        
        return df_clean
    
    def normalize_features(self, df, exclude_columns=['riesgo']):
        """
        Normaliza las variables para que estén en la misma escala.
        
        ¿Qué es normalizar?
        Convertir todos los valores a una escala similar (generalmente entre -1 y 1)
        """
        df_normalized = df.copy()
        
        # Seleccionar solo columnas numéricas
        numeric_cols = df_normalized.select_dtypes(include=[np.number]).columns
        
        # Quitar las que no queremos normalizar
        cols_to_normalize = [col for col in numeric_cols if col not in exclude_columns]
        
        if len(cols_to_normalize) > 0:
            # Aplicar StandardScaler
            df_normalized[cols_to_normalize] = self.scaler.fit_transform(df_normalized[cols_to_normalize])
            print(f"  ➜ Normalizadas {len(cols_to_normalize)} columnas")
        
        return df_normalized
    
    def get_cleaning_summary(self):
        """
        Retorna un resumen de todo lo que se limpió.
        """
        return self.cleaning_report


# Pruebas del módulo
if __name__ == "__main__":
    # Creamos datos de prueba con MUCHOS problemas a propósito
    data = {
        'promedio_actual': [85, 90, None, 70, 150, "80", "noventa"],
        'asistencia_clases': [95, 80, 75, None, 88, "Si", "100"],
        'tareas_entregadas': [100, 90, 85, 95, 92, "Alto", 88],
        'participacion_clase': [80, 70, None, 60, 75, "Bajo", 85],
        'horas_estudio': [20, 15, 25, -5, 30, "10", 200],
        'promedio_evaluaciones': [88, 92, 85, 78, 90, 85, 88],
        'cursos_reprobados': [0, 1, 2, 0, 1, "0", 1],
        'actividades_extracurriculares': ["['deportes']", "['club', 'musica']", "[]", 0, 2, "Muchas", 3],
        'reportes_disciplinarios': [0, 0, 1, 0, 0, "No", 1],
        'riesgo': ["no riesgo", "riesgo", "riesgo", "no riesgo", 0, "Si", 1]
    }
    
    df_test = pd.DataFrame(data)
    
    print("=" * 60)
    print("DATOS ORIGINALES (CON PROBLEMAS):")
    print("=" * 60)
    print(df_test)
    print(f"\n Valores faltantes totales: {df_test.isnull().sum().sum()}")
    print(f" Tipos de datos:")
    print(df_test.dtypes)
    
    # Aplicar limpieza
    cleaner = DataCleaner()
    df_cleaned = cleaner.clean_data(df_test)
    
    print("\n" + "=" * 60)
    print("DATOS DESPUÉS DE LA LIMPIEZA:")
    print("=" * 60)
    print(df_cleaned)
    print(f"\n Valores faltantes después: {df_cleaned.isnull().sum().sum()}")
    print(f" Tipos de datos después:")
    print(df_cleaned.dtypes)
    
    print("\n" + "=" * 60)
    print("RESUMEN DE LIMPIEZA:")
    print("=" * 60)
    for key, value in cleaner.get_cleaning_summary().items():
        print(f"  • {key}: {value}")