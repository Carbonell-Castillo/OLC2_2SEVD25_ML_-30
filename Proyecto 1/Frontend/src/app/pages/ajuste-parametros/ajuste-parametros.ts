// ajuste-parametros.ts
import { Component, ChangeDetectorRef, OnInit } from '@angular/core';
import { CommonModule, DecimalPipe } from '@angular/common';
import { FormsModule } from '@angular/forms'; 
import { DataService, TrainResponse, Hyperparameters } from '../../services/data.service'; 
// Asumo que tu data.service.ts está en '../../services/data.service'

@Component({
 selector: 'app-ajuste-parametros',
 standalone: true, // Debe ser standalone
 imports: [CommonModule, DecimalPipe, FormsModule], // Importar módulos necesarios
 templateUrl: './ajuste-parametros.html',
 styleUrl: './ajuste-parametros.css',
})
export class AjusteParametros implements OnInit {

    // 1. Parámetros del modelo (Inicializados con valores por defecto/último entrenamiento conocido)
    hyperparams: Hyperparameters = {
        max_iter: 1000,
        C: 0.5,
        solver: 'lbfgs'
    };
    
    // 2. Estado de la interfaz
    isTraining: boolean = false;
    errorMessage: string | null = null;
    successMessage: string | null = null;
    
    // 3. Resultados
    lastMetrics: TrainResponse['metrics'] | null = null;
    
 
    dataReadyForTraining: boolean = false; 


   constructor(
        private dataService: DataService,
        private cdr: ChangeDetectorRef 
    ) {}

    ngOnInit(): void {

    }

    /**
     * Inicia el entrenamiento del modelo con los parámetros actuales del formulario.
     */
    trainModelWithCustomParams(): void {
        this.errorMessage = null;
        this.successMessage = null;
        this.isTraining = true;
        this.cdr.detectChanges(); // Forzar la actualización del botón a 'Entrenando...'

        console.log('Iniciando entrenamiento con:', this.hyperparams);

        this.dataService.trainModelWithParams(this.hyperparams).subscribe({
            next: (response: TrainResponse) => {
                this.isTraining = false;
                this.lastMetrics = response.metrics;
                this.successMessage = response.message;
                
                // Actualizar los parámetros de la UI con los que realmente se usaron (por si el backend los modifica)
                this.hyperparams = response.metrics.hyperparams_used; 
                
                this.cdr.detectChanges();
            },
            error: (err) => {
                this.isTraining = false;
                const errorMsg = err.error?.error || 'Error desconocido al reentrenar. Asegúrate que los datos estén cargados y limpios.';
                this.errorMessage = errorMsg;
                this.cdr.detectChanges();
            }
        });
    }

    /**
     * Función para ayudar a formatear el F1-Score
     * @param metric Valor de la métrica (entre 0 y 1)
     * @returns Cadena con el formato de porcentaje.
     */
    formatMetric(metric: number | undefined): string {
        if (metric === undefined || metric === null) return 'N/A';
        return (metric * 100).toFixed(2) + '%';
    }
}