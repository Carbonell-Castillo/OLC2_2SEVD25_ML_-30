import pandas as pd
import os

class DataLoader:
    
    
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
    
    # Columnas que deben ser numéricas 
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
       
        self.upload_folder = upload_folder
        
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
            print(f" Carpeta '{upload_folder}' creada")
    
    def load_csv(self, file_path):
        
        try:
            df = pd.read_csv(file_path)
            
            if df.empty:
                return None, "El archivo CSV está vacío"
            
            missing_columns = self.validate_columns(df)
            if missing_columns:
                return None, f"Faltan las siguientes columnas: {', '.join(missing_columns)}"
            
            print(f" CSV cargado exitosamente: {df.shape[0]} filas, {df.shape[1]} columnas")
            return df, None
            
        except FileNotFoundError:
            return None, "Archivo no encontrado"
        except pd.errors.EmptyDataError:
            return None, "El archivo está vacío"
        except Exception as e:
            return None, f"Error al cargar CSV: {str(e)}"
    
    def validate_columns(self, df):
        
        df_columns = set(df.columns)              
        required_columns = set(self.REQUIRED_COLUMNS)  
        
        missing = required_columns - df_columns
        
        return list(missing)
    
    def get_data_info(self, df):
       
        return {
            'total_rows': len(df),                          # Cantidad de estudiantes
            'total_columns': len(df.columns),               # Cantidad de variables
            'columns': list(df.columns),                    # Nombres de las columnas
            'missing_values': df.isnull().sum().to_dict(),  # Valores faltantes por columna
            'data_types': df.dtypes.astype(str).to_dict()   # Tipo de dato de cada columna
        }
    
    def validate_numeric_columns(self, df):
       
        errors = []
        
        for col in self.NUMERIC_COLUMNS:
            if col in df.columns:
                try:
                    numeric_values = pd.to_numeric(df[col], errors='coerce')
                    
                    original_nulls = df[col].isnull().sum()
                    new_nulls = numeric_values.isnull().sum()
                    non_numeric_count = new_nulls - original_nulls
                    
                    if non_numeric_count > 0:
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


if __name__ == "__main__":
    loader = DataLoader()
    
    # Intentar cargar un CSV de ejemplo
    df, error = loader.load_csv("datos_prueba.csv")
    
    if error:
        print(f" Error: {error}")
    else:
        print(" Datos cargados correctamente")
        info = loader.get_data_info(df)
        print(f"Total de estudiantes: {info['total_rows']}")
        print(f"\nValores faltantes por columna:")
        for col, missing in info['missing_values'].items():
            if missing > 0:
                print(f"  - {col}: {missing} valores faltantes")