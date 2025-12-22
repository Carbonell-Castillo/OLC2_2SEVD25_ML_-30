import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';


export interface UploadResponse {
  filename: string;
  message: string;
  info: {
    total_rows: number;
    total_columns: number;
    columns: string[];
    data_types: { [key: string]: string };
    missing_values: { [key: string]: number };
  };
  preview: { [key: string]: any }[];
}

// Nota: Puedes definir interfaces más detalladas para CleanResponse y TrainResponse
export interface CleanResponse {
  message: string;
  summary: {
    duplicates_removed: number;
    missing_values_handled: number;
    text_converted_to_numeric: { [key: string]: string };
    values_adjusted: number; // Agregado para el caso de 'horas_estudio: 2 valores ajustados'
  };
  cleaned_info: {
    total_rows: number;
    total_columns: number;
    columns: string[];
    missing_values: { [key: string]: number };
    preview: { [key: string]: any }[];
  };
}
export interface TrainResponse {
    message: string;
    metrics: {
        accuracy: number;
        f1_score: number;
        precision: number;
        recall: number;
        n_train: number;
        n_test: number;
        model_path: string;
        hyperparams_used: {
            C: number;
            max_iter: number;
            solver: string;
        };
    };
}
export interface Hyperparameters {
    max_iter: number;
    C: number;
    solver: string;
}
export interface EvaluationMetricsResponse {
    message: string;
    metrics: TrainResponse['metrics']; // Reutilizamos la estructura de métricas
    confusion_matrix: {
        true_positives: number;
        false_positives: number;
        false_negatives: number;
        true_negatives: number;
    };
}

export interface PredictionInput {
    promedio_actual: number;
    asistencia_clases: number;
    tareas_entregadas: number;
    participacion_clase: number;
    horas_estudio: number;
   actividades_extracurriculares: number;
    cursos_reprobados: number;
    reportes_disciplinarios: number;
        promedio_evaluaciones: number;
}

export interface PredictionResult {
    prediction: 0 | 1; // 0 para 'no riesgo', 1 para 'riesgo'
    prediction_meaning: 'no riesgo' | 'riesgo';
    probability_riesgo: number; // Probabilidad de ser riesgo
}

export interface PredictionResponse {
    message: string;
    result: PredictionResult;
}

// --- SERVICIO DE DATOS ---

@Injectable({
  providedIn: 'root',
})
export class DataService {
  private apiUrl = 'http://localhost:5000/api'; // Asegúrate de que esta URL sea correcta

  constructor(private http: HttpClient) {}

  healthCheck(): Observable<any> {
    return this.http.get(`${this.apiUrl}/health`);
  }

  uploadFile(file: File): Observable<UploadResponse> {
    const formData = new FormData();
    formData.append('file', file, file.name);
    return this.http.post<UploadResponse>(`${this.apiUrl}/upload`, formData);
  }

  cleanData(): Observable<CleanResponse> {
    // Asume que el backend ya tiene el archivo en memoria/disco
    return this.http.post<CleanResponse>(`${this.apiUrl}/clean`, {});
  }

  trainModel(): Observable<TrainResponse> {
    // Asume que el backend ya tiene los datos limpios
    return this.http.post<TrainResponse>(`${this.apiUrl}/train`, {});
  }

  resetSystem(): Observable<any> {
    return this.http.post(`${this.apiUrl}/reset`, {});
  }

  trainModelWithParams(params: Hyperparameters): Observable<TrainResponse> {
    // POST /api/train_with_params
  return this.http.post<TrainResponse>(`${this.apiUrl}/train_with_params`, params);
 }
 getEvaluationMetrics(): Observable<EvaluationMetricsResponse> {
        return this.http.get<EvaluationMetricsResponse>(`${this.apiUrl}/get_metrics`);
    }
    predictRisk(data: PredictionInput): Observable<PredictionResponse> {
        // El endpoint es /api/predict (POST)
        return this.http.post<PredictionResponse>(`${this.apiUrl}/predict`, data);
    }
 
}