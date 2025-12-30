 
import numpy as np

class KMeans:
    """
    Implementación de K-Means desde cero
    
    Parámetros:
    -----------
    n_clusters : int, default=3
        Número de clusters a formar
    max_iterations : int, default=100
        Número máximo de iteraciones
    random_state : int, default=None
        Semilla para reproducibilidad
    """
    
    def __init__(self, n_clusters=3, max_iterations=100, random_state=None):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.inertia_ = None
        
    def fit(self, X):
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        #  Inicializar centroides aleatoriamente
        random_indices = np.random.choice(len(X), self.n_clusters, replace=False)
        self.centroids = X[random_indices].copy()
        
        for iteration in range(self.max_iterations):
            #  Asignar cada punto al centroide más cercano
            old_labels = self.labels
            distances = self._calculate_distances(X)
            self.labels = np.argmin(distances, axis=1)
            
            #  Recalcular centroides
            new_centroids = np.zeros_like(self.centroids)
            for k in range(self.n_clusters):
                cluster_points = X[self.labels == k]
                if len(cluster_points) > 0:
                    new_centroids[k] = cluster_points.mean(axis=0)
                else:
                    # Si un cluster queda vacío, reinicializar con un punto aleatorio
                    new_centroids[k] = X[np.random.choice(len(X))]
            
            #  Verificar convergencia
            if old_labels is not None and np.array_equal(old_labels, self.labels):
                print(f" Convergencia en iteración {iteration + 1}")
                break
            
            self.centroids = new_centroids
        
        # Calcular inercia
        self._calculate_inertia(X)
        
        return self
    
    def predict(self, X):
        
        distances = self._calculate_distances(X)
        return np.argmin(distances, axis=1)
    
    def _calculate_distances(self, X):
        
        distances = np.zeros((len(X), self.n_clusters))
        for k in range(self.n_clusters):
            distances[:, k] = np.linalg.norm(X - self.centroids[k], axis=1)
        return distances
    
    def _calculate_inertia(self, X):
        """
        Calcular suma de distancias cuadradas dentro de cada cluster
        """
        self.inertia_ = 0
        for k in range(self.n_clusters):
            cluster_points = X[self.labels == k]
            if len(cluster_points) > 0:
                self.inertia_ += np.sum((cluster_points - self.centroids[k]) ** 2)
    
    def fit_predict(self, X):
        """
        Ajustar y predecir en un solo paso
        """
        self.fit(X)
        return self.labels