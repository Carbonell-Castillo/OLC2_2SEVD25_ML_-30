import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-configuracion-modelo',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './configuracion-modelo.html'
})
export class ConfiguracionModeloComponent {
  // Parámetros del modelo
  numClusters: number = 5;
  maxIteraciones: number = 300;
  algoritmo: string = 'K-Means++';
  metrica: string = 'Euclidiana';

  // Estados del entrenamiento
  isTraining: boolean = false;
  progress: number = 0;
  trainingComplete: boolean = false;

  iniciarEntrenamiento() {
    this.isTraining = true;
    this.trainingComplete = false;
    this.progress = 0;

    const interval = setInterval(() => {
      this.progress += Math.floor(Math.random() * 15) + 5;
      if (this.progress >= 100) {
        this.progress = 100;
        this.isTraining = false;
        this.trainingComplete = true;
        clearInterval(interval);
      }
    }, 400);
  }
}