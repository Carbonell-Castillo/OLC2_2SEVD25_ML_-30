from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
# Asegúrate de que estas importaciones sean correctas
from src.data.data_loader import DataLoader
from src.data.data_cleaner import DataCleaner
from src.ml.training import train_model
from src.ml.prediction import predict_risk

import pandas as pd
import numpy as np


app = Flask(__name__)

CORS(app) 

# CONFIGURACIÓN DE SUBIDA DE ARCHIVOS

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Máximo 16MB

# VARIABLES GLOBALES PARA GUARDAR DATOS EN MEMORIA

current_data = None
cleaned_data = None
last_metrics_results = None
# INICIALIZAR NUESTRAS CLASES

loader = DataLoader(UPLOAD_FOLDER)
cleaner = DataCleaner()

# FUNCIONES AUXILIARES

def allowed_file(filename):
    """
    Verifica que el archivo tenga extensión .csv
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_to_serializable(obj):
    """
    Convierte tipos numpy a tipos nativos de Python para JSON
    """
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif pd.isna(obj):
        return None
    # Maneja tipos numéricos de NumPy (int, float)
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif hasattr(obj, 'item'):  # otros tipos numpy
        return obj.item()
    return obj

# ============================================================================
# ENDPOINTS DE LA API
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Endpoint para verificar que el servidor esté funcionando
    """
    return jsonify({
        'status': 'ok',
        'message': 'StudentGuard API está funcionando correctamente',
        'version': '1.0.0'
    }), 200


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """
    Recibe un archivo CSV desde el frontend y lo procesa
    """
    global current_data
    
    # Verificar que se envió un archivo
    if 'file' not in request.files:
        return jsonify({'error': 'No se envió ningún archivo'}), 400
    
    file = request.files['file']
    
    # Verificar que se seleccionó un archivo
    if file.filename == '':
        return jsonify({'error': 'No se seleccionó ningún archivo'}), 400
    
    # Verificar que sea un CSV
    if not allowed_file(file.filename):
        return jsonify({'error': 'Solo se permiten archivos CSV'}), 400
    
    try:
        # Guardar el archivo de forma segura
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
    
        # Cargar y validar el CSV
        df, error = loader.load_csv(filepath)
        
        if error:
            # Si hubo error, eliminamos el archivo
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': error}), 400
        
        # Guardar los datos en memoria
        current_data = df
        
        # Obtener información del dataset
        info = loader.get_data_info(df)
        
        # Preparar preview (primeras 10 filas)
        preview_data = df.head(10).copy()
        # Reemplazar NaN con None (que se traduce a null en JSON)
        preview_data = preview_data.where(pd.notnull(preview_data), None) 
        preview_dict = preview_data.to_dict('records')
        
        # Convertir info y preview a tipos serializables (soluciona el error de NumPy/JSON)
        preview_dict = convert_to_serializable(preview_dict)
        info = convert_to_serializable(info) 
        
        return jsonify({
            'message': 'Archivo cargado exitosamente',
            'filename': filename,
            'info': info,
            'preview': preview_dict
        }), 200
    
    except Exception as e:
        # En caso de error, imprime el traceback en la consola del backend para diagnóstico
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error al procesar archivo: {str(e)}'}), 500


@app.route('/api/clean', methods=['POST'])
def clean_data():
    """
    Limpia los datos que fueron cargados previamente
    """
    global current_data, cleaned_data
    
    # Verificar que haya datos cargados
    if current_data is None:
        return jsonify({
            'error': 'No hay datos cargados. Primero sube un archivo CSV'
        }), 400
    
    try:
        print("\n" + "=" * 60)
        print("INICIANDO PROCESO DE LIMPIEZA")
        print("=" * 60)
        
        # Limpiar los datos
        cleaned_data = cleaner.clean_data(current_data)
        
        # Obtener resumen de limpieza
        summary = cleaner.get_cleaning_summary()
        
        # Preparar preview de datos limpios
        preview_data = cleaned_data.head(10).copy()
        preview_data = preview_data.where(pd.notnull(preview_data), None)
        preview_dict = preview_data.to_dict('records')
        
        # Convertir a tipos serializables
        preview_dict = convert_to_serializable(preview_dict)
        summary = convert_to_serializable(summary)
        
        # Calcular valores faltantes después de limpieza
        missing_values = {}
        for col in cleaned_data.columns:
            missing_values[col] = int(cleaned_data[col].isnull().sum())
        
        print("\n" + "=" * 60)
        print("LIMPIEZA COMPLETADA")
        print("=" * 60)
        
        return jsonify({
            'message': 'Datos limpiados exitosamente',
            'summary': summary,
            'cleaned_info': {
                'total_rows': int(len(cleaned_data)),
                'total_columns': int(len(cleaned_data.columns)),
                'missing_values': missing_values,
                'columns': list(cleaned_data.columns),
                'preview': preview_dict
            }
        }), 200
    
    except Exception as e:
        print(f"\n ERROR EN LIMPIEZA: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error al limpiar datos: {str(e)}'}), 500

@app.route('/api/train', methods=['POST'])
def train():
    """
    Se entrena el modelo de riesgo usando los datos limpios actuales.
    Devuelve las métricas principales (accuracy, precision, recall, f1).
    """
    global cleaned_data, last_metrics_results

    # 1) Verificar que ya haya datos limpios
    if cleaned_data is None:
        return jsonify({
            "error": "No hay datos limpios. Primero Limpia los datos."
        }), 400

    try:
        print("\n" + "=" * 60)
        print("INICIANDO ENTRENAMIENTO DEL MODELO")
        print("=" * 60)

        # 2) Llamar a la función de entrenamiento
        metrics = train_model(cleaned_data)
        # Guardar las métricas y la matriz de confusión globalmente
        last_metrics_results = metrics

        print("\n Entrenamiento completado")
        print(f"   - Accuracy:  {metrics['accuracy']:.3f}")
        print(f"   - Precision: {metrics['precision']:.3f}")
        print(f"   - Recall:    {metrics['recall']:.3f}")
        print(f"   - F1-score:  {metrics['f1_score']:.3f}")
        print(f"   - Modelo guardado en: {metrics['model_path']}")

        # 3) Devolver métricas al frontend
        metrics = convert_to_serializable(metrics)
        
        return jsonify({
            "message": "Modelo entrenado exitosamente",
            "metrics": metrics
        }), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": f"Error al entrenar modelo: {str(e)}"
        }), 500
    

@app.route('/api/train_with_params', methods=['POST'])
def train_with_params():
    """
    Entrena el modelo con hiperparámetros personalizados.
    """
    global cleaned_data, last_metrics_results

    # Verificar que haya datos limpios
    if cleaned_data is None:
        return jsonify({
            "error": "No hay datos limpios. Primero Limpia los datos."
        }), 400

    try:
        print("\n" + "=" * 60)
        print("ENTRENAMIENTO CON HIPERPARÁMETROS PERSONALIZADOS")
        print("=" * 60)
        
        # Obtener hiperparámetros del body (si no se envían, usa defaults)
        data = request.get_json() or {}
        
        hyperparams = {
            'max_iter': data.get('max_iter', 1000),
            'C': data.get('C', 0.5),
            'solver': data.get('solver', 'lbfgs')
        }
        
        print(f"\n Hiperparámetros recibidos:")
        print(f"   - max_iter: {hyperparams['max_iter']}")
        print(f"   - C: {hyperparams['C']}")
        print(f"   - solver: {hyperparams['solver']}")
        
        # Importar función de entrenamiento con parámetros
        from src.ml.training import train_model_with_params
        
        # Entrenar con parámetros personalizados
        metrics = train_model_with_params(cleaned_data, hyperparams)
        # Guardar las métricas y la matriz de confusión globalmente
        last_metrics_results = metrics
        metrics = convert_to_serializable(metrics)

        return jsonify({
            "message": "Modelo entrenado con hiperparámetros personalizados",
            "metrics": metrics
        }), 200

    except ValueError as e:
        # Errores de validación
        return jsonify({
            "error": str(e)
        }), 400

    except Exception as e:
        print(f"\n ERROR EN ENTRENAMIENTO: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": f"Error al entrenar modelo: {str(e)}"
        }), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Recibe los datos de UN estudiante y devuelve la predicción de riesgo.
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "error": "Se requiere un cuerpo JSON con los datos del estudiante."
            }), 400

        result = predict_risk(data)

        result = convert_to_serializable(result)

        return jsonify({
            "message": "Predicción generada correctamente",
            "result": result
        }), 200

    except ValueError as e:
        # Errores de validación (campos faltantes, tipos, modelo no entrenado)
        return jsonify({
            "error": str(e)
        }), 400

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": f"Error interno al generar la predicción: {str(e)}"
        }), 500


@app.route('/api/data/info', methods=['GET'])
def get_data_info():
    """
    Obtiene información detallada sobre los datos actuales
    """
    global current_data, cleaned_data
    
    if current_data is None:
        return jsonify({'error': 'No hay datos cargados'}), 400
    
    # Usar datos limpios si existen, sino usar los originales
    data_to_use = cleaned_data if cleaned_data is not None else current_data
    
    try:
        # Información básica
        info = loader.get_data_info(data_to_use)
        
        # Estadísticas descriptivas
        stats_df = data_to_use.describe()
        statistics = {}
        
        for col in stats_df.columns:
            statistics[col] = {}
            for stat_name in stats_df.index:
                value = stats_df.loc[stat_name, col]
                statistics[col][stat_name] = float(value) if pd.notnull(value) else None
        
        # Marcar si los datos ya fueron limpiados
        info['is_cleaned'] = cleaned_data is not None
        info['statistics'] = statistics
        
        # Convertir a tipos serializables
        info = convert_to_serializable(info)
        
        return jsonify(info), 200
    
    except Exception as e:
        return jsonify({'error': f'Error al obtener información: {str(e)}'}), 500


@app.route('/api/data/export', methods=['GET'])
def export_cleaned_data():
    """
    Exporta los datos limpios como un nuevo archivo CSV
    """
    global cleaned_data
    
    if cleaned_data is None:
        return jsonify({
            'error': 'No hay datos limpios disponibles. Primero limpia los datos'
        }), 400
    
    try:
        # Guardar CSV limpio
        export_filename = 'datos_limpios.csv'
        export_path = os.path.join(app.config['UPLOAD_FOLDER'], export_filename)
        cleaned_data.to_csv(export_path, index=False)
        
        return jsonify({
            'message': 'Datos exportados exitosamente',
            'filename': export_filename,
            'path': export_path,
            'rows': int(len(cleaned_data))
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Error al exportar datos: {str(e)}'}), 500


@app.route('/api/data/compare', methods=['GET'])
def compare_data():
    """
    Compara datos originales con datos limpios
    """
    global current_data, cleaned_data
    
    if current_data is None:
        return jsonify({'error': 'No hay datos cargados'}), 400
    
    if cleaned_data is None:
        return jsonify({'error': 'No hay datos limpios. Primero limpia los datos'}), 400
    
    try:
        comparison = {
            'original': {
                'rows': int(len(current_data)),
                'missing_values': int(current_data.isnull().sum().sum())
            },
            'cleaned': {
                'rows': int(len(cleaned_data)),
                'missing_values': int(cleaned_data.isnull().sum().sum())
            },
            'changes': {
                'rows_removed': int(len(current_data) - len(cleaned_data)),
                'missing_values_fixed': int(current_data.isnull().sum().sum() - cleaned_data.isnull().sum().sum())
            }
        }
        
        return jsonify(comparison), 200
    
    except Exception as e:
        return jsonify({'error': f'Error al comparar datos: {str(e)}'}), 500


@app.route('/api/get_metrics', methods=['GET'])
def get_metrics():
    """
    Devuelve las métricas del último entrenamiento, incluyendo la matriz de confusión.
    """
    global last_metrics_results
    print("last_metrics_results content:", last_metrics_results) # Mantener para depuración
    
    # 1) Verificar que se haya entrenado el modelo al menos una vez
    if last_metrics_results is None:
        return jsonify({
            "error": "Modelo no entrenado. Primero entrena un modelo usando /api/train o /api/train_with_params."
        }), 404

    try:
     
        response_data = {
            "message": "Métricas de evaluación cargadas correctamente.",
            "metrics": last_metrics_results,  # El objeto de métricas
            "confusion_matrix": last_metrics_results.get('confusion_matrix', {

                "true_positives": 0,
                "false_positives": 0,
                "false_negatives": 0,
                "true_negatives": 0,
            })
        }

        # 3) Asegurar la serialización
        response_data = convert_to_serializable(response_data)
        
        f1_score = response_data['metrics'].get('f1_score', 0.0) # Usar .get() por seguridad
        
        print("\n" + "=" * 60)
        print(" MÉTRICAS DE EVALUACIÓN DEVUELTAS")
        print(f" - F1 Score: {f1_score:.3f}") # ¡CORREGIDO!
        print("=" * 60)
        
        return jsonify(response_data), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": f"Error al recuperar métricas: {str(e)}"
        }), 500
@app.route('/api/reset', methods=['POST'])
def reset_data():
    """
    Reinicia todo el sistema
    """
    global current_data, cleaned_data, last_metrics_results
    
    # Limpiar variables globales
    current_data = None
    cleaned_data = None
    last_metrics_results = None
    
    # Limpiar archivos temporales
    try:
        for file in os.listdir(app.config['UPLOAD_FOLDER']):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file)
            if os.path.isfile(file_path):
                os.unlink(file_path)
        
        print("\n Sistema reiniciado correctamente\n")
        
        return jsonify({
            'message': 'Sistema reiniciado correctamente'
        }), 200
    
    except Exception as e:
        return jsonify({
            'error': f'Error al reiniciar sistema: {str(e)}'
        }), 500
    
    

# ============================================================================
# MANEJO DE ERRORES
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    """
    Maneja endpoints no encontrados
    """
    return jsonify({
        'error': 'Endpoint no encontrado',
        'message': 'Verifica la ruta de la petición'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """
    Maneja errores internos del servidor
    """
    return jsonify({
        'error': 'Error interno del servidor',
        'message': 'Contacta al administrador si el error persiste'
    }), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    """
    Maneja archivos demasiado grandes
    """
    return jsonify({
        'error': 'Archivo demasiado grande',
        'message': 'El archivo debe ser menor a 16MB'
    }), 413


# ============================================================================
# INICIAR EL SERVIDOR
# ============================================================================

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    
    print("\n" + "=" * 60)
    print(" StudentGuard Backend v1.0.0")
    print("=" * 60)
    print("\n Endpoints disponibles:")
    print("   GET  /api/health            - Verificar estado del servidor")
    print("   POST /api/upload            - Cargar archivo CSV")
    print("   POST /api/clean             - Limpiar datos cargados")
    print("   GET  /api/data/info         - Información de los datos")
    print("   GET  /api/data/compare      - Comparar datos originales vs limpios")
    print("   GET  /api/data/export       - Exportar datos limpios")
    print("   POST /api/reset             - Reiniciar el sistema")
    print("   POST /api/train             - Entrenar modelo de riesgo")
    print("   POST /api/train_with_params - Entrenar modelo (hiperparámetros personalizados)")
    print("   POST /api/predict           - Predecir riesgo de un estudiante")

    print("\n Servidor corriendo en: http://localhost:5000")
    print("=" * 60)
    
    # Iniciar servidor
    app.run(debug=True, port=5000, host='0.0.0.0')