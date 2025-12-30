"""
Módulo para clustering de texto usando TF-IDF
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans as SKLearnKMeans
import numpy as np


class TextClusterer:
    """
    Realiza clustering de texto usando TF-IDF + K-Means
    """
    
    def __init__(self, n_clusters=3, max_features=100):
        """
        Args:
            n_clusters: Número de clusters de reseñas
            max_features: Máximo número de palabras a considerar
        """
        self.n_clusters = n_clusters
        self.max_features = max_features
        
        # Configurar TF-IDF
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,    # Top 100 palabras más importantes
            min_df=2,                     # Palabra debe aparecer en al menos 2 docs
            max_df=0.8,                   # Ignorar si aparece en más del 80%
            ngram_range=(1, 2),           # Considerar 1 palabra y 2 palabras juntas
            strip_accents='unicode',       # Quitar acentos
            lowercase=True                 # Convertir a minúsculas
        )
        
        # Configurar K-Means (usamos sklearn por simplicidad)
        self.kmeans = SKLearnKMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,
            max_iter=300
        )
    
    def fit_predict(self, texts):
        """
        Entrenar TF-IDF + K-Means y obtener labels
        
        Args:
            texts: Lista de textos (reseñas limpias)
            
        Returns:
            labels: Array con el cluster de cada texto
        """
        # Limpiar textos vacíos
        texts_clean = [str(t) if t and str(t).strip() else "" for t in texts]
        
        # PASO 1: Vectorización TF-IDF
        print(f" Vectorizando {len(texts_clean)} reseñas con TF-IDF...")
        X_tfidf = self.vectorizer.fit_transform(texts_clean)
        print(f"   Matriz TF-IDF: {X_tfidf.shape} (reseñas x palabras)")
        
        # PASO 2: K-Means sobre vectores
        print(f" Clustering de texto con K={self.n_clusters}...")
        labels = self.kmeans.fit_predict(X_tfidf)
        
        return labels
    
    def get_top_terms(self, n_terms=5):
        """
        Obtener las palabras más importantes de cada cluster
        
        Args:
            n_terms: Número de palabras top por cluster
            
        Returns:
            dict: {cluster_id: [palabra1, palabra2, ...]}
        """
        # Obtener nombres de features (palabras)
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Ordenar centroides por importancia
        order_centroids = self.kmeans.cluster_centers_.argsort()[:, ::-1]
        
        # Extraer top palabras por cluster
        top_terms = {}
        for i in range(self.n_clusters):
            top_terms[i] = [
                feature_names[ind] for ind in order_centroids[i, :n_terms]
            ]
        
        return top_terms
    
    def get_cluster_stats(self, labels):
        """
        Calcular estadísticas de cada cluster
        
        Returns:
            list: Estadísticas por cluster
        """
        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)
        
        stats = []
        for cluster_id, count in zip(unique, counts):
            stats.append({
                "cluster_id": int(cluster_id),
                "size": int(count),
                "percentage": round((count / total) * 100, 2)
            })
        
        return stats