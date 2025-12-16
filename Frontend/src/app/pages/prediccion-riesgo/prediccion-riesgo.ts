// prediccion-riesgo.ts

import { Component, ElementRef, ViewChild, OnInit, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms'; // Necesario para ngModel
import { DataService, PredictionInput, PredictionResponse, PredictionResult } from '../../services/data.service'; // Importar servicio e interfaces

@Component({
 selector: 'app-prediccion-riesgo',
 standalone: true,
 imports: [CommonModule, FormsModule], // ¡Añadir FormsModule!
 templateUrl: './prediccion-riesgo.html',
 styleUrl: './prediccion-riesgo.css'
})
export class PrediccionRiesgo implements OnInit { // Implementar OnInit


    predictionInput: PredictionInput = {
        promedio_actual: 0, // 0.0 - 10.0
        asistencia_clases: 0, // 0.0 - 100.0
        tareas_entregadas: 0, // 0.0 - 100.0
        participacion_clase: 0, // 0 - 10
        horas_estudio: 0, // Horas
        cursos_reprobados: 0,
        actividades_extracurriculares: 0,
        reportes_disciplinarios: 0,
        promedio_evaluaciones: 0
    };

    
    // Estado y Referencias
    isLoading: boolean = false;
    errorMessage: string | null = null;
  showPredictionResult = true; 
  showRiskDisplay = false;   
  riskIconClass: string = '';
  riskTextClass: string = '';
  riskTextContent: string = '';
  riskDetailContent: string = '';
  riskCardClass: string = 'border-accent-cyan'; 

  @ViewChild('riskCard') riskCard!: ElementRef;

  constructor(
        private dataService: DataService,
        private cdr: ChangeDetectorRef 
    ) {}
    
    ngOnInit(): void {
        // Inicializar el mensaje de error/éxito
    }
    

  onPredictRisk() {
        this.isLoading = true;
        this.errorMessage = null;
        this.showPredictionResult = false; // Asumimos que algo va a pasar
        
        console.log('Datos enviados para predicción:', this.predictionInput);
        // **VALIDACIÓN SIMPLE (Opcional pero Recomendada)**
        if (this.predictionInput.promedio_actual < 0 || this.predictionInput.asistencia_clases > 100) {
            this.errorMessage = 'Por favor, ingrese valores válidos (ej: Asistencia debe ser entre 0 y 100).';
            this.isLoading = false;
            this.showPredictionResult = true;
            return;
        }

        this.dataService.predictRisk(this.predictionInput).subscribe({
            next: (response: PredictionResponse) => {
                console.log('Respuesta de predicción recibida:', response);
                this.isLoading = false;
                this.updateRiskDisplay(response.result);
                this.cdr.detectChanges();
            },
            error: (err) => {
                this.isLoading = false;
                this.showRiskDisplay = false;
                this.showPredictionResult = true; // Mostrar mensaje de error
                console.error('Error al predecir:', err);
                this.errorMessage = err.error?.error || 'Error de conexión. Asegúrate de que el modelo haya sido entrenado.';
            this.cdr.detectChanges();
              }
        });
  }
    
    /**
     * Actualiza la tarjeta de resultado con la predicción real del modelo.
     */
    updateRiskDisplay(result: PredictionResult): void {
        this.showPredictionResult = false;
        this.showRiskDisplay = true;
        const probability = result.probability_riesgo * 100;
        console.log('Actualizando visualización de riesgo con:', result);
        if (result.prediction === 1) { // Riesgo
            // Alto Riesgo (Estilo Peligro - Rojo)
            console.log('Configurando visualización para ALTO RIESGO');
            this.riskIconClass = 'fas fa-exclamation-triangle text-8xl mb-4 text-red-500';
            this.riskTextClass = 'text-3xl font-extrabold text-red-400';
            this.riskTextContent = 'RIESGO ALTO';
            this.riskDetailContent = `Probabilidad de Riesgo: ${probability.toFixed(2)}%. Se recomienda intervención.`;
            this.riskCardClass = 'border-red-500'; // Borde rojo
        } else {
            // Bajo Riesgo (Estilo Éxito - Verde)
            this.riskIconClass = 'fas fa-shield-alt text-8xl mb-4 text-green-500';
            this.riskTextClass = 'text-3xl font-extrabold text-green-400';
            this.riskTextContent = 'RIESGO BAJO';
            this.riskDetailContent = `Probabilidad de Riesgo: ${(100 - probability).toFixed(2)}% (Éxito). Se proyecta un rendimiento óptimo.`;
            this.riskCardClass = 'border-accent-cyan'; // Borde cian
        }
    }
    
    /**
     * Limpia los campos del formulario y restablece el resultado.
     */
    onCancel() {
        // Resetear modelo
        this.predictionInput = {
            promedio_actual: 0,
            asistencia_clases: 0,
            tareas_entregadas: 0,
            participacion_clase: 0,
            horas_estudio: 0,
            cursos_reprobados: 0,
            actividades_extracurriculares: 0,
            reportes_disciplinarios: 0,
             promedio_evaluaciones: 0,
        };
        // Resetear vista
        this.showPredictionResult = true;
        this.showRiskDisplay = false;
        this.errorMessage = null;
        this.riskCardClass = 'border-accent-cyan'; 
    }
}