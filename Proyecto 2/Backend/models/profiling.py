import numpy as np
import pandas as pd


class ClusterProfiler:
    """
    Genera perfiles e interpretaciones comprensibles de los clusters
    """

    def __init__(self, df, cluster_stats, review_clustering=None):
        """
        Args:
            df: DataFrame con los datos y la columna 'cluster'
            cluster_stats: estadísticas numéricas por cluster
            review_clustering: resultados del clustering de texto (opcional)
        """
        self.df = df
        self.cluster_stats = cluster_stats
        self.review_clustering = review_clustering

    def generate_profiles(self):
        """
        Generar perfiles completos de cada cluster
        """
        profiles = []

        # Valores globales para comparación (alto / bajo)
        global_means = self._calculate_global_means()

        for cluster in self.cluster_stats:
            cluster_id = cluster["cluster_id"]
            cluster_df = self.df[self.df["cluster"] == cluster_id]

            description = self._build_description(
                cluster, cluster_df, global_means, cluster_id
            )

            profile = {
                "cluster_id": cluster_id,
                "size": cluster["size"],
                "percentage": cluster["percentage"],
                "canal_principal": cluster["canal_principal"],
                "characteristics": cluster["characteristics"],
                "review_topics": self._get_review_topics(cluster_id),
                "description": description
            }

            profiles.append(profile)

        return profiles

    # ===================== MÉTODOS INTERNOS =====================

    def _calculate_global_means(self):
        """
        Calcular promedios globales para comparación
        """
        numeric_cols = [
            'frecuencia_compra',
            'monto_total_gastado',
            'monto_promedio_compra',
            'dias_desde_ultima_compra',
            'antiguedad_cliente_meses',
            'numero_productos_distintos',
            'longitud_reseña'
        ]

        return {
            col: self.df[col].mean()
            for col in numeric_cols
            if col in self.df.columns
        }

    def _qualitative_level(self, value, global_mean):
        """
        Convertir un valor numérico en etiqueta cualitativa
        """
        if global_mean == 0 or global_mean is None:
            return "desconocido"

        if value >= global_mean * 1.2:
            return "alto"
        elif value <= global_mean * 0.8:
            return "bajo"
        else:
            return "medio"

    def _get_review_topics(self, cluster_id):
        """
        Obtener temas predominantes de reseñas por cluster
        """
        if not self.review_clustering:
            return []

        return self.review_clustering.get("top_keywords", {}).get(cluster_id, [])

    def _build_description(self, cluster, cluster_df, global_means, cluster_id):
        """
        Construir descripción en lenguaje humano del cluster
        """
        char = cluster["characteristics"]

        freq_level = self._qualitative_level(
            char.get("frecuencia_compra", 0),
            global_means.get("frecuencia_compra", 0)
        )

        gasto_level = self._qualitative_level(
            char.get("monto_total_gastado", 0),
            global_means.get("monto_total_gastado", 0)
        )

        recency_level = self._qualitative_level(
            char.get("dias_desde_ultima_compra", 0),
            global_means.get("dias_desde_ultima_compra", 0)
        )

        canal = cluster.get("canal_principal", "N/A")
        topics = self._get_review_topics(cluster_id)

        description = (
            f"Este segmento agrupa clientes con una frecuencia de compra {freq_level} "
            f"y un gasto total {gasto_level}. "
            f"Los clientes utilizan principalmente el canal {canal}. "
        )

        if recency_level == "bajo":
            description += "Se trata de clientes activos con compras recientes. "
        elif recency_level == "alto":
            description += "Incluye clientes con periodos largos sin realizar compras. "

        if topics:
            description += (
                "En sus reseñas predominan temas relacionados con "
                f"{', '.join(topics)}."
            )

        return description.strip()
