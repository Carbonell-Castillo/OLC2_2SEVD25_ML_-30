import { Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';

// --- Interfaces para el tipado de datos ---
export interface UploadResponse {
  status: string;
  message: string;
  file_id: string;
  rows: number;
  columns: string[];
}

export interface TrainRequest {
  file_id: string;
  n_clusters: number;
  max_iterations?: number;
  random_state?: number;
}

export interface ClusteringResults {
  status: string;
  file_id: string;
  summary: {
    n_clusters: number;
    overall_quality: string;
    recommendation: string;
  };
  metrics: {
    silhouette_score: number;
    davies_bouldin_score: number;
  };
  clusters: Array<{
    cluster_id: number;
    size: number;
    percentage: number;
    description: string;
    main_channel: string;
  }>;
}

@Injectable({
  providedIn: 'root'
})
export class DataService {
  // Cambia esto a la URL de tu servidor FastAPI en producción si es necesario
  private readonly API_URL = 'http://localhost:8000';

  constructor(private http: HttpClient) { }

  /**
   * 1. Sube un archivo CSV o Excel al backend
   */
  uploadFile(file: File): Observable<UploadResponse> {
    const formData = new FormData();
    formData.append('file', file);
    return this.http.post<UploadResponse>(`${this.API_URL}/upload`, formData);
  }

  /**
   * 2. Inicia el entrenamiento del modelo K-Means
   */
  trainModel(params: TrainRequest): Observable<ClusteringResults> {
    return this.http.post<ClusteringResults>(`${this.API_URL}/train`, params);
  }

  /**
   * 3. Obtiene los resultados de un entrenamiento previo
   */
  getResults(fileId: string): Observable<ClusteringResults> {
    return this.http.get<ClusteringResults>(`${this.API_URL}/results/${fileId}`);
  }

  /**
   * 4. Genera y obtiene los datos de la visualización PCA
   */
  getVisualization(fileId: string): Observable<any> {
    return this.http.get(`${this.API_URL}/visualization/${fileId}`);
  }

  /**
   * 5. Descarga el archivo CSV con los resultados (Blob)
   */
  downloadCSV(fileId: string): Observable<Blob> {
    return this.http.get(`${this.API_URL}/download/${fileId}`, {
      responseType: 'blob'
    });
  }

  /**
   * 6. Exporta el reporte en formato PDF
   */
  exportPDF(fileId: string): Observable<any> {
    return this.http.get(`${this.API_URL}/export/pdf/${fileId}`);
  }

  /**
   * 7. Verifica el estado de salud de la API
   */
  checkHealth(): Observable<any> {
    return this.http.get(`${this.API_URL}/health`);
  }
}