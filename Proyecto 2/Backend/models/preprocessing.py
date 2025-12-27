import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats
import re

class DataPreprocessor:
    """
    Clase para preprocesar datos de clientes y reseñas
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []
        self.channel_mapping = {}
        self.original_data = None
    
    def load_and_clean(self, file_path):
        """
        Cargar y limpiar datos del CSV
        """
        # CARGAR CSV
        df = pd.read_csv(file_path)
        self.original_data = df.copy()
        
        print(f" Datos cargados: {len(df)} filas, {len(df.columns)} columnas")
        
        # CALCULAR LONGITUD DE RESEÑA
        if 'texto_reseña' in df.columns:
            if 'longitud_reseña' not in df.columns:
                df['longitud_reseña'] = df['texto_reseña'].apply(
                    lambda x: len(str(x).split()) if pd.notna(x) else 0
                )
        
        #  LIMPIAR TEXTO DE RESEÑAS 
        if 'texto_reseña' in df.columns:
            print(" Limpiando texto de reseñas...")
            df['texto_limpio'] = df['texto_reseña'].apply(self.clean_review_text)
            cleaned_count = df['texto_limpio'].notna().sum()
            print(f" {cleaned_count} reseñas limpiadas")
        
        # ELIMINAR DUPLICADOS POR CLIENTE
        if 'cliente_id' in df.columns:
            df = df.groupby('cliente_id').first().reset_index()
            print(f" Datos únicos por cliente: {len(df)} filas")
        
        # MANEJO DE VALORES NULOS 
        df = self._handle_missing_values(df)
        
        # ELIMINAR OUTLIERS 
        df = self._remove_outliers(df)
        
        #  ONE-HOT ENCODING DEL CANAL
        df = self._encode_categorical(df)
        
        # PREPARAR MATRIZ DE FEATURES Y NORMALIZAR
        X_scaled, feature_names = self._prepare_features(df)
        
        print(f" Preprocesamiento completo: {X_scaled.shape}")
        
        return X_scaled, df, feature_names
    
    def _handle_missing_values(self, df):
        """
        Manejar valores faltantes con estrategia híbrida
        MEJORADO: Limpia automáticamente valores no numéricos
        """
        
        # LIMPIAR VALORES NO NUMÉRICOS 
        
        # Definir columnas que DEBEN ser numéricas
        numeric_columns = [
            'frecuencia_compra',
            'monto_total_gastado', 
            'monto_promedio_compra',
            'dias_desde_ultima_compra',
            'antiguedad_cliente_meses',
            'numero_productos_distintos'
        ]
        
        print(f" Limpiando valores no numéricos en columnas numéricas...")
        
        # Convertir cada columna a numérico, forzando errores a NaN
        for col in numeric_columns:
            if col in df.columns:
                # Intentar convertir a numérico
                df_temp = pd.to_numeric(df[col], errors='coerce')
                
                # Contar cuántos valores se convirtieron a NaN (eran no numéricos)
                original_nans = df[col].isna().sum()
                new_nans = df_temp.isna().sum()
                non_numeric_count = new_nans - original_nans
                
                # Reemplazar la columna con la versión limpia
                df[col] = df_temp
                
                if non_numeric_count > 0:
                    print(f"     '{col}': {non_numeric_count} valores no numéricos convertidos a NaN")
        
        # ELIMINAR FILAS CON NaN EN COLUMNAS CRÍTICAS 
        
        # Columnas críticas (si tienen NaN, eliminar la fila completa)
        critical_cols = ['frecuencia_compra', 'monto_total_gastado', 'canal_principal']
        
        # Contar filas antes
        rows_before = len(df)
        
        # Eliminar filas con NaN en columnas críticas
        df = df.dropna(subset=critical_cols)
        
        # Contar filas eliminadas
        rows_after = len(df)
        rows_removed = rows_before - rows_after
        
        if rows_removed > 0:
            print(f"  {rows_removed} filas eliminadas por valores nulos en columnas críticas")
        
        # RELLENAR COLUMNAS OPCIONALES CON MEDIANA
        
        # Columnas opcionales (si tienen NaN, rellenar con mediana)
        optional_cols = [
            'dias_desde_ultima_compra', 
            'antiguedad_cliente_meses', 
            'numero_productos_distintos',
            'monto_promedio_compra'
        ]
        
        for col in optional_cols:
            if col in df.columns and df[col].isna().any():
                # Calcular mediana (ignora NaN automáticamente)
                median_value = df[col].median()
                
                # Contar valores a rellenar
                null_count = df[col].isna().sum()
                
                # Crear copia para evitar SettingWithCopyWarning
                df = df.copy()
                
                # Rellenar con mediana
                df[col] = df[col].fillna(median_value)
                
                print(f" Rellenados {null_count} valores en '{col}' con mediana: {median_value:.2f}")
        
        return df
    
    def _remove_outliers(self, df):
        """
        Eliminar outliers usando Z-score
        
        Z-score mide cuántas desviaciones estándar está un valor del promedio:
        - Z-score < 3: Valor normal (99.7% de datos)
        - Z-score ≥ 3: Outlier (0.3% de datos extremos)
        """
        numeric_cols = [
            'frecuencia_compra', 
            'monto_total_gastado',
            'monto_promedio_compra', 
            'dias_desde_ultima_compra',
            'antiguedad_cliente_meses', 
            'numero_productos_distintos'
        ]
        
        # Verificar que existan las columnas
        numeric_cols = [col for col in numeric_cols if col in df.columns]
        
        if not numeric_cols:
            return df
        
        initial_len = len(df)
        
        try:
            # Calcular Z-scores (valores absolutos)
            z_scores = np.abs(stats.zscore(df[numeric_cols]))
            
            # Mantener solo filas donde TODOS los z-scores < 3
            # (3 desviaciones estándar = 99.7% de datos normales)
            df_clean = df[(z_scores < 3).all(axis=1)]
            
            outliers_removed = initial_len - len(df_clean)
            
            if outliers_removed > 0:
                percentage = (outliers_removed / initial_len) * 100
                print(f"  {outliers_removed} outliers eliminados ({percentage:.1f}%)")
            else:
                print(f" No se detectaron outliers")
            
            return df_clean
            
        except Exception as e:
            print(f"  No se pudo detectar outliers: {e}")
            return df
    
    def _encode_categorical(self, df):
        """One-hot encoding para canal_principal"""
        if 'canal_principal' not in df.columns:
            return df
        
        unique_channels = df['canal_principal'].unique()
        
        # Crear columnas binarias para cada canal
        for channel in unique_channels:
            # Limpiar nombre del canal para evitar problemas
            clean_channel_name = str(channel).replace('_', '').replace(' ', '').replace('í', 'i').replace('ó', 'o')
            df[f'canal_{clean_channel_name}'] = (df['canal_principal'] == channel).astype(int)
        
        self.channel_mapping = {i: ch for i, ch in enumerate(unique_channels)}
        
        print(f" Canales encontrados: {list(unique_channels)}")
        
        return df
    
    def _prepare_features(self, df):
        """Seleccionar y normalizar features para clustering"""
        numeric_cols = [
            'frecuencia_compra', 
            'monto_total_gastado',
            'monto_promedio_compra', 
            'dias_desde_ultima_compra',
            'antiguedad_cliente_meses', 
            'numero_productos_distintos'
        ]

        # ELIMINAR cliente_id antes del clustering
        # solo se usa para agrupar, no para entrenar
        if 'cliente_id' in df.columns:
            print(f" Eliminando 'cliente_id'")
            df = df.drop(columns=['cliente_id'])

        # Verificar que existan las columnas
        numeric_cols = [col for col in numeric_cols if col in df.columns]
        
        # Buscar columnas de canales (empiezan con 'canal_' pero NO es 'canal_principal')
        channel_cols = [col for col in df.columns if col.startswith('canal_') and col != 'canal_principal']
        
        all_features = numeric_cols + channel_cols
        
        print(f" Features seleccionadas: {all_features}")
        
        # Extraer valores y convertir a float
        X = df[all_features].values.astype(float)
        
        # Normalizar (CRÍTICO para K-Means)
        X_scaled = self.scaler.fit_transform(X)
        
        self.feature_names = all_features
        
        return X_scaled, all_features
    
    def clean_review_text(self, text):
        """
        Limpieza de texto de reseñas
        
        Pasos:
        1. Convertir a minúsculas
        2. Eliminar caracteres especiales
        3. Eliminar stopwords (palabras comunes sin significado)
        4. Filtrar palabras muy cortas
        """
        if pd.isna(text):
            return ""
        
        # Convertir a minúsculas
        text = str(text).lower()
        
        # Eliminar caracteres especiales (mantener solo letras y espacios)
        text = re.sub(r'[^a-záéíóúñ\s]', '', text)
        
        # Stopwords en español (palabras comunes sin significado)
        stopwords = [
            'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'una', 
            'es', 'por', 'para', 'con', 'no', 'se', 'los', 'las',
            'del', 'al', 'lo', 'como', 'más', 'pero', 'sus', 'le',
            'ya', 'o', 'fue', 'este', 'ha', 'sí', 'porque', 'esta',
            'son', 'entre', 'está', 'cuando', 'muy', 'sin', 'sobre',
            'ser', 'tiene', 'también', 'me', 'hasta', 'hay', 'donde',
            'han', 'quien', 'están', 'estado', 'desde', 'todo', 'nos'
        ]
        
        # Tokenizar y filtrar
        words = [w for w in text.split() if w not in stopwords and len(w) > 2]
        
        return ' '.join(words)
    
    # ========== MÉTODO NUEVO (AL MISMO NIVEL QUE clean_review_text) ==========
    def calculate_cluster_stats(self, df, labels):
        """
        Calcular estadísticas por cluster
        
        Args:
            df: DataFrame con los datos
            labels: Array con las etiquetas de cluster
            
        Returns:
            list: Lista de diccionarios con estadísticas por cluster
        """
        import numpy as np
        
        df_temp = df.copy()
        df_temp['cluster'] = labels
        
        stats = []
        
        for cluster_id in sorted(df_temp['cluster'].unique()):
            cluster_data = df_temp[df_temp['cluster'] == cluster_id]
            
            # Características numéricas promedio
            characteristics = {}
            numeric_cols = [
                'frecuencia_compra',
                'monto_total_gastado',
                'monto_promedio_compra',
                'dias_desde_ultima_compra'
            ]
            
            for col in numeric_cols:
                if col in cluster_data.columns:
                    characteristics[col] = round(cluster_data[col].mean(), 2)
            
            # Canal principal (moda)
            canal_principal = "N/A"
            if 'canal_principal' in cluster_data.columns:
                canal_principal = cluster_data['canal_principal'].mode()[0] if len(cluster_data['canal_principal'].mode()) > 0 else "N/A"
            
            stats.append({
                "cluster_id": int(cluster_id),
                "size": int(len(cluster_data)),
                "percentage": round((len(cluster_data) / len(df_temp)) * 100, 2),
                "characteristics": characteristics,
                "canal_principal": canal_principal
            })
        
        return stats