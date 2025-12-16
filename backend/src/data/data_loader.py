import pandas as pd
import os

class DataLoader:
    """
    Esta clase se encarga de cargar los archivos CSV que suben las instituciones
    y valida que tengan todas las columnas necesarias para entrenar el modelo.
    
    Básicamente hace dos cosas importantes:
    1. Verificar que el CSV tenga todas las columnas que necesitamos
    2. Cargar los datos en memoria para poder trabajar con ellos
    """
    
    # Estas son las columnas que DEBE tener el CSV según el documento del proyecto
    # Si falta alguna, no podemos entrenar el modelo correctamente
    # NOTA: Pueden existir columnas extras (como carnet, nombres, etc.) pero estas son obligatorias
    REQUIRED_COLUMNS = [
        'promedio_actual',                    # Promedio del estudiante (0-100)
        'asistencia_clases',                  # % de asistencia (0-100)
        'tareas_entregadas',                  # % de tareas entregadas (0-100)
        'participacion_clase',                # Nivel de participación (0-100)
        'horas_estudio',                      # Horas que estudia por semana
        'promedio_evaluaciones',              # Promedio en exámenes parciales
        'cursos_reprobados',                  # Cantidad de cursos que ha reprobado
        'actividades_extracurriculares',      # Cuántas actividades extra tiene
        'reportes_disciplinarios',            # Reportes de mala conducta
        'riesgo'                              # Lo que queremos predecir (0 = no riesgo, 1 = riesgo)
    ]
    
    # Columnas que DEBEN ser numéricas (las demás pueden ser cualquier cosa)
    NUMERIC_COLUMNS = [
        'promedio_actual',
        'asistencia_clases',
        'tareas_entregadas',
        'participacion_clase',
        'horas_estudio',
        'promedio_evaluaciones',
        'cursos_reprobados',
        'actividades_extracurriculares',
        'reportes_disciplinarios'
    ]
    
    def __init__(self, upload_folder='uploads'):
        """
        Cuando creamos el objeto, le decimos dónde guardar los archivos temporales
        """
        self.upload_folder = upload_folder
        
        # Si la carpeta no existe, la creamos automáticamente
        # Esto evita errores cuando intentemos guardar archivos
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
            print(f"📁 Carpeta '{upload_folder}' creada")
    
    def load_csv(self, file_path):
        """
        Esta función carga el archivo CSV y verifica que esté bien estructurado
        
        ¿Por qué hacemos esto?
        - Evitamos que el sistema se caiga si el CSV está mal
        - Le avisamos al usuario si falta alguna columna importante
        - Validamos que no esté vacío
        
        Args:
            file_path: La ruta donde está guardado el CSV
            
        Returns:
            Una tupla con (DataFrame, error)
            - Si todo salió bien: (datos, None)
            - Si hubo problema: (None, "mensaje de error")
        """
        try:
            # Leemos el CSV con pandas (librería para manejar datos tabulares)
            df = pd.read_csv(file_path)
            
            # Verificar que no esté vacío
            # Un CSV vacío no nos sirve para entrenar
            if df.empty:
                return None, "El archivo CSV está vacío"
            
            # Verificar que tenga todas las columnas necesarias
            missing_columns = self.validate_columns(df)
            if missing_columns:
                return None, f"Faltan las siguientes columnas: {', '.join(missing_columns)}"
            
            # Si llegamos aquí, todo está bien 🎉
            print(f"✅ CSV cargado exitosamente: {df.shape[0]} filas, {df.shape[1]} columnas")
            return df, None
            
        except FileNotFoundError:
            # El archivo no existe en esa ruta
            return None, "Archivo no encontrado"
        except pd.errors.EmptyDataError:
            # El CSV literalmente no tiene nada
            return None, "El archivo está vacío"
        except Exception as e:
            # Cualquier otro error raro que pueda pasar
            return None, f"Error al cargar CSV: {str(e)}"
    
    def validate_columns(self, df):
        """
        Compara las columnas del CSV con las que necesitamos
        
        ¿Por qué es importante?
        Si falta una columna, el modelo no puede entrenar correctamente.
        Por ejemplo, si falta "asistencia_clases", no podemos saber
        si esa variable afecta el riesgo de deserción.
        
        Args:
            df: El DataFrame con los datos cargados
            
        Returns:
            Lista de columnas que faltan (vacía si están todas)
        """
        # Convertimos a conjuntos (sets) para hacer la comparación más fácil
        df_columns = set(df.columns)              # Columnas que tiene el CSV
        required_columns = set(self.REQUIRED_COLUMNS)  # Columnas que necesitamos
        
        # Restamos para ver cuáles faltan
        missing = required_columns - df_columns
        
        return list(missing)
    
    def get_data_info(self, df):
        """
        Obtiene un resumen de los datos para mostrárselo al usuario
        
        Esto es útil porque:
        - El usuario puede ver cuántos estudiantes tiene
        - Puede identificar si hay valores faltantes
        - Puede verificar que los tipos de datos sean correctos
        
        Args:
            df: DataFrame con los datos
            
        Returns:
            Diccionario con información del dataset
        """
        return {
            'total_rows': len(df),                          # Cantidad de estudiantes
            'total_columns': len(df.columns),               # Cantidad de variables
            'columns': list(df.columns),                    # Nombres de las columnas
            'missing_values': df.isnull().sum().to_dict(),  # Valores faltantes por columna
            'data_types': df.dtypes.astype(str).to_dict()   # Tipo de dato de cada columna
        }
    
    def validate_numeric_columns(self, df):
        """
        Valida que las columnas que deben ser numéricas realmente lo sean
        
        ¿Por qué es importante?
        Si una columna numérica contiene texto (ej: "Si" en lugar de 1),
        el proceso de limpieza fallará al intentar hacer operaciones matemáticas.
        
        Args:
            df: DataFrame a validar
            
        Returns:
            String con mensaje de error, o None si todo está bien
        """
        errors = []
        
        for col in self.NUMERIC_COLUMNS:
            if col in df.columns:
                # Intentar convertir a numérico
                try:
                    # Convertir forzosamente, si hay texto se convertirá a NaN
                    numeric_values = pd.to_numeric(df[col], errors='coerce')
                    
                    # Contar cuántos valores NO se pudieron convertir (excluyendo los que ya eran NaN)
                    original_nulls = df[col].isnull().sum()
                    new_nulls = numeric_values.isnull().sum()
                    non_numeric_count = new_nulls - original_nulls
                    
                    if non_numeric_count > 0:
                        # Mostrar algunos ejemplos de valores problemáticos
                        problematic_values = df[col][~df[col].isnull() & numeric_values.isnull()].unique()[:3]
                        errors.append(
                            f"'{col}' tiene {non_numeric_count} valores no numéricos. "
                            f"Ejemplos: {list(problematic_values)}"
                        )
                except Exception as e:
                    errors.append(f"'{col}': Error al validar - {str(e)}")
        
        if errors:
            return "; ".join(errors)
        return None


# Esta parte solo se ejecuta si corremos este archivo directamente
# Sirve para hacer pruebas rápidas
if __name__ == "__main__":
    loader = DataLoader()
    
    # Intentar cargar un CSV de ejemplo
    df, error = loader.load_csv("datos_prueba.csv")
    
    if error:
        print(f"❌ Error: {error}")
    else:
        print("✅ Datos cargados correctamente")
        info = loader.get_data_info(df)
        print(f"Total de estudiantes: {info['total_rows']}")
        print(f"\nValores faltantes por columna:")
        for col, missing in info['missing_values'].items():
            if missing > 0:
                print(f"  - {col}: {missing} valores faltantes")