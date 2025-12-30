 
import numpy as np
from models.kmeans import KMeans

print("=" * 50)
print(" PRUEBA BÁSICA DE K-MEANS")
print("=" * 50)

# Generar datos sintéticos
print("\n Generando datos sintéticos...")
np.random.seed(42)

cluster1 = np.random.randn(30, 2) + [2, 2]
cluster2 = np.random.randn(30, 2) + [-2, -2]
cluster3 = np.random.randn(30, 2) + [2, -2]

X = np.vstack([cluster1, cluster2, cluster3])
print(f" Datos generados: {X.shape[0]} puntos, {X.shape[1]} dimensiones")

# Entrenar K-Means
print("\n Entrenando K-Means con 3 clusters...")
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)

# Mostrar resultados
print(f"\n RESULTADOS:")
print(f"   Inercia: {kmeans.inertia_:.2f}")
print(f"   Clusters únicos: {np.unique(labels)}")
print(f"   Tamaños: {np.bincount(labels)}")

print("\n" + "=" * 50)
print(" PRUEBA EXITOSA")
print("=" * 50)