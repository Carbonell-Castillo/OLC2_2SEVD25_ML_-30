from models.preprocessing import DataPreprocessor
from models.kmeans import KMeans
import pandas as pd

print("=" * 60)
print(" PRUEBA DE PREPROCESAMIENTO Y CLUSTERING")
print("=" * 60)

#  Cargar y preprocesar datos
print("\n Cargando datos...")
preprocessor = DataPreprocessor()
X_scaled, df, feature_names = preprocessor.load_and_clean('data/uploads/test_data.csv')

print(f"\n Features utilizadas ({len(feature_names)}):")
for i, feat in enumerate(feature_names, 1):
    print(f"   {i}. {feat}")

# Entrenar K-Means
print("\n Entrenando K-Means con 4 clusters...")
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X_scaled)

# Agregar labels al DataFrame
df['cluster'] = labels

# Análisis de clusters
print("\n RESULTADOS DEL CLUSTERING:")
print(f"   Inercia: {kmeans.inertia_:.2f}")
print(f"   Clusters encontrados: {sorted(set(labels))}")

print("\n Distribución de clusters:")
for cluster_id in sorted(set(labels)):
    count = sum(labels == cluster_id)
    percentage = (count / len(labels)) * 100
    print(f"   Cluster {cluster_id}: {count} clientes ({percentage:.1f}%)")

#  Características promedio por cluster
print("\n Características promedio por cluster:")
numeric_features = ['frecuencia_compra', 'monto_total_gastado', 
                   'monto_promedio_compra', 'dias_desde_ultima_compra']

for cluster_id in sorted(set(labels)):
    print(f"\n   🔹 CLUSTER {cluster_id}:")
    cluster_data = df[df['cluster'] == cluster_id]
    
    for feat in numeric_features:
        if feat in df.columns:
            avg = cluster_data[feat].mean()
            print(f"      {feat}: {avg:.2f}")
    
    # Canal más común
    if 'canal_principal' in df.columns:
        top_channel = cluster_data['canal_principal'].mode()
        if len(top_channel) > 0:
            print(f"      Canal principal: {top_channel.iloc[0]}")

print("\n" + "=" * 60)
print(" ¡PRUEBA COMPLETA EXITOSA!")
print("=" * 60)