"""
Módulo para calcular métricas de evaluación de clustering
"""
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)


class ClusteringMetrics:
    """
    Clase para calcular e interpretar métricas de clustering
    """
    
    @staticmethod
    def calculate_all_metrics(X, labels):
        """
        Calcular todas las métricas de evaluación
        
        Args:
            X: Matriz de features normalizadas (numpy array)
            labels: Etiquetas de cluster (numpy array)
            
        Returns:
            dict: Diccionario con métricas e interpretación
        """
        # Validar que hay al menos 2 clusters
        n_clusters = len(set(labels))
        if n_clusters < 2:
            return {
                "error": "Se necesitan al menos 2 clusters para calcular métricas"
            }
        
        metrics = {}
        
        try:
            # 1. Silhouette Score (rango: -1 a 1, mayor es mejor)
            # Mide qué tan similar es un objeto a su propio cluster
            # vs otros clusters
            metrics['silhouette_score'] = float(silhouette_score(X, labels))
            
            # 2. Calinski-Harabasz Index (mayor es mejor)
            # Ratio de dispersión entre-clusters vs dentro-clusters
            metrics['calinski_harabasz_score'] = float(
                calinski_harabasz_score(X, labels)
            )
            
            # 3. Davies-Bouldin Index (menor es mejor)
            # Promedio de similitud entre cada cluster y su más similar
            metrics['davies_bouldin_score'] = float(
                davies_bouldin_score(X, labels)
            )
            
            # 4. Interpretación automática
            metrics['interpretation'] = ClusteringMetrics._interpret_metrics(metrics)
            
        except Exception as e:
            metrics['error'] = str(e)
        
        return metrics
    
    @staticmethod
    def _interpret_metrics(metrics):
        """
        Interpretar las métricas y dar recomendaciones
        
        Args:
            metrics: Diccionario con las métricas calculadas
            
        Returns:
            dict: Interpretación y recomendaciones
        """
        silhouette = metrics['silhouette_score']
        calinski = metrics['calinski_harabasz_score']
        davies = metrics['davies_bouldin_score']
        
        # Interpretar Silhouette Score
        if silhouette > 0.7:
            quality = "Excelente"
            silhouette_desc = "Clusters muy bien definidos y fuertemente separados"
        elif silhouette > 0.5:
            quality = "Bueno"
            silhouette_desc = "Clusters razonablemente separados con buena cohesión"
        elif silhouette > 0.3:
            quality = "Aceptable"
            silhouette_desc = "Estructura de clusters presente pero con solapamiento"
        elif silhouette > 0:
            quality = "Pobre"
            silhouette_desc = "Clusters débilmente definidos con mucho solapamiento"
        else:
            quality = "Muy Pobre"
            silhouette_desc = "Clustering incorrecto, objetos asignados a clusters equivocados"
        
        # Interpretar Davies-Bouldin
        if davies < 0.5:
            davies_desc = "Excelente separación entre clusters"
        elif davies < 1.0:
            davies_desc = "Buena separación entre clusters"
        elif davies < 1.5:
            davies_desc = "Separación aceptable, algunos clusters pueden solaparse"
        else:
            davies_desc = "Pobre separación, clusters muy similares entre sí"
        
        # Generar recomendación
        if silhouette < 0.3:
            recommendation = "Se recomienda probar con diferente número de clusters o revisar el preprocesamiento de datos"
        elif silhouette < 0.5:
            recommendation = "El clustering es aceptable pero puede mejorarse experimentando con diferentes valores de K"
        else:
            recommendation = "El número de clusters es apropiado para estos datos"
        
        # Determinar si vale la pena el clustering
        clustering_worthwhile = silhouette > 0.25
        
        return {
            "overall_quality": quality,
            "silhouette_interpretation": silhouette_desc,
            "davies_bouldin_interpretation": davies_desc,
            "calinski_harabasz_note": f"Valor alto ({calinski:.2f}) indica buena separación" if calinski > 100 else f"Valor bajo ({calinski:.2f}) sugiere clusters poco diferenciados",
            "recommendation": recommendation,
            "clustering_worthwhile": clustering_worthwhile
        }
    
    @staticmethod
    def get_metric_ranges():
        """
        Obtener rangos y descripciones de cada métrica
        
        Returns:
            dict: Información sobre cada métrica
        """
        return {
            "silhouette_score": {
                "range": "[-1, 1]",
                "best": "Más cercano a 1",
                "interpretation": {
                    "0.7 - 1.0": "Excelente",
                    "0.5 - 0.7": "Bueno",
                    "0.3 - 0.5": "Aceptable",
                    "0.0 - 0.3": "Pobre",
                    "-1.0 - 0.0": "Muy pobre (incorrecta asignación)"
                },
                "description": "Mide la cohesión intra-cluster y separación inter-cluster"
            },
            "calinski_harabasz_score": {
                "range": "[0, ∞)",
                "best": "Valores más altos",
                "interpretation": {
                    "> 500": "Excelente separación",
                    "200 - 500": "Buena separación",
                    "100 - 200": "Separación moderada",
                    "< 100": "Pobre separación"
                },
                "description": "Ratio de dispersión entre-clusters vs dentro-clusters (Variance Ratio Criterion)"
            },
            "davies_bouldin_score": {
                "range": "[0, ∞)",
                "best": "Más cercano a 0",
                "interpretation": {
                    "< 0.5": "Excelente",
                    "0.5 - 1.0": "Bueno",
                    "1.0 - 1.5": "Aceptable",
                    "> 1.5": "Pobre"
                },
                "description": "Promedio de similitu    d entre cada cluster y su cluster más similar"
            }
        }