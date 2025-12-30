import { Component, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { DataService, TrainRequest, ClusteringResults } from '../../services/data.service';

@Component({
  selector: 'app-configuracion-modelo',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './configuracion-modelo.html'
})
export class ConfiguracionModeloComponent {
  // Parámetros del modelo (Vinculados a ngModel)
  numClusters: number = 3;
  maxIteraciones: number = 100;
  algoritmo: string = 'K-Means++'; // Nota: Tu backend actual usa K-Means estándar
  metrica: string = 'Euclidiana';

  // Estados
  isTraining: boolean = false;
  progress: number = 0;
  trainingComplete: boolean = false;
  
  // Resultados del backend
  resultados: ClusteringResults | null = null;

  constructor(
    private dataService: DataService,
    private cdr: ChangeDetectorRef
  ) {}

  iniciarEntrenamiento() {
    // 1. Obtener el ID del último archivo subido (puedes guardarlo en localStorage o un servicio)
    const fileId = localStorage.getItem('lastFileId'); 

    if (!fileId) {
      alert('No se encontró un archivo cargado. Por favor, sube uno primero.');
      return;
    }

    this.isTraining = true;
    this.trainingComplete = false;
    this.progress = 10; // Progreso visual inicial

    // Preparar los datos para el backend
    const requestData: TrainRequest = {
      file_id: fileId,
      n_clusters: this.numClusters,
      max_iterations: this.maxIteraciones,
      random_state: 42
    };

    console.log('Iniciando entrenamiento con datos:', requestData);
    // 2. Iniciar simulación de progreso mientras esperamos la respuesta
    const interval = setInterval(() => {
      if (this.progress < 90) {
        this.progress += Math.floor(Math.random() * 5);
        this.cdr.detectChanges();
      }
    }, 500);

    // 3. Llamada Real a la API
    this.dataService.trainModel(requestData).subscribe({
      next: (res: ClusteringResults) => {
        clearInterval(interval);
        this.progress = 100;
        this.resultados = res;
        this.isTraining = false;
        this.trainingComplete = true;
        
        console.log('Entrenamiento exitoso:', res);
        this.cdr.detectChanges();
        
        // Opcional: Notificar a otros componentes que hay resultados listos
        localStorage.setItem('clusteringResults', JSON.stringify(res));
      },
      error: (err) => {
        clearInterval(interval);
        this.isTraining = false;
        this.progress = 0;
        alert('Error en el entrenamiento: ' + (err.error?.detail || 'Error interno'));
        this.cdr.detectChanges();
      }
    });
  }
}