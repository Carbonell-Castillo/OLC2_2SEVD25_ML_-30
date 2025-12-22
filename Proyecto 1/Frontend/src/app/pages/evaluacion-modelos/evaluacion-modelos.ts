// evaluacion-modelos.ts
import { Component, OnInit, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { DataService, EvaluationMetricsResponse } from '../../services/data.service'; 
@Component({
 selector: 'app-evaluacion-modelos',
 standalone: true,
 imports: [CommonModule], // DecimalPipe para formatear porcentajes
 templateUrl: './evaluacion-modelos.html',
 styleUrl: './evaluacion-modelos.css',
})
export class EvaluacionModelos implements OnInit {

    // 1. Estado para almacenar los datos de evaluación
    evaluationData: EvaluationMetricsResponse | null = null;
    
    // 2. Estado de la interfaz
    isLoading: boolean = false;
    errorMessage: string | null = null;
    successMessage: string | null = null; // No es estrictamente necesario, pero útil
    
   constructor(
        private dataService: DataService,
        private cdr: ChangeDetectorRef 
    ) {}

    ngOnInit(): void {
        this.loadEvaluationMetrics();
    }

    /**
     * Carga las métricas del último modelo entrenado desde el backend.
     */
    loadEvaluationMetrics(): void {
        this.isLoading = true;
        this.errorMessage = null;
        this.evaluationData = null; // Limpiar datos anteriores

        this.dataService.getEvaluationMetrics().subscribe({
            next: (response: EvaluationMetricsResponse) => {
                this.isLoading = false;
                this.evaluationData = response;
                this.successMessage = response.message || 'Métricas cargadas correctamente.';
                this.cdr.detectChanges();
            },
            error: (err) => {
                this.isLoading = false;
                console.error('Error al cargar métricas:', err);
                const defaultMsg = 'Asegúrate de haber cargado datos y entrenado un modelo previamente.';
                this.errorMessage = err.error?.error || `Error: ${defaultMsg}`;
                this.cdr.detectChanges();
            }
        });
    }

    
    formatMetric(metric: number | undefined): string {
        if (metric === undefined || metric === null) return 'N/A';
        return (metric * 100)
            .toFixed(2)
            + '%';
    }
    

    exportReport(): void {
        alert('Funcionalidad de exportación de reporte detallado (Aún no implementada en el backend).');
        // Aquí se llamaría a un dataService.exportReport()
    }
}