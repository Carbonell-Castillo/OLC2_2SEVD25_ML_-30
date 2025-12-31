import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_test_data(n_clients=200, n_reviews=500):
    np.random.seed(42)
    print(" Generando datos sintéticos...")
    
    canales = ['web', 'móvil', 'tienda_física']
    categorias = ['Electrónica', 'Ropa', 'Hogar', 'Deportes', 'Libros', 'Juguetes']
    
    clientes = []
    for i in range(1, n_clients + 1):
        cliente = {
            'cliente_id': i,
            'frecuencia_compra': np.random.randint(1, 50),
            'monto_total_gastado': round(np.random.uniform(100, 10000), 2),
            'monto_promedio_compra': round(np.random.uniform(50, 500), 2),
            'dias_desde_ultima_compra': np.random.randint(1, 365),
            'antiguedad_cliente_meses': np.random.randint(1, 60),
            'canal_principal': np.random.choice(canales),
            'numero_productos_distintos': np.random.randint(1, 20),
        }
        clientes.append(cliente)
    
    reseñas_textos = [
        "Excelente producto muy buena calidad",
        "No me gustó esperaba más por el precio",
        "Cumple con lo prometido satisfecho",
        "Muy mal servicio no lo recomiendo",
        "Increíble superó mis expectativas",
        "Regular podría mejorar",
        "Muy buena relación calidad precio",
        "Decepcionante no volvería a comprar",
        "Producto defectuoso",
        "Perfecto justo lo que necesitaba"
    ]
    
    reseñas = []
    for i in range(1, n_reviews + 1):
        dias_atras = np.random.randint(1, 365)
        fecha = datetime.now() - timedelta(days=dias_atras)
        reseña = {
            'reseña_id': i,
            'cliente_id': np.random.randint(1, n_clients + 1),
            'texto_reseña': np.random.choice(reseñas_textos),
            'fecha_reseña': fecha.strftime('%Y-%m-%d'),
            'producto_categoria': np.random.choice(categorias),
        }
        reseñas.append(reseña)
    
    df_clientes = pd.DataFrame(clientes)
    df_reseñas = pd.DataFrame(reseñas)
    df_completo = pd.merge(df_clientes, df_reseñas, on='cliente_id', how='left')
    
    output_path = 'data/uploads/test_data.csv'
    df_completo.to_csv(output_path, index=False)
    
    print(f" Datos generados: {output_path}")
    print(f"   Clientes: {n_clients}, Reseñas: {n_reviews}, Total: {len(df_completo)}")
    return df_completo

if __name__ == '__main__':
    df = generate_test_data(n_clients=800, n_reviews=2000)
    print('\n Primeras 5 filas:')
    print(df.head())