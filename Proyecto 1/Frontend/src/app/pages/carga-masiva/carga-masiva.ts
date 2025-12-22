import { Component, OnInit, ChangeDetectorRef } from '@angular/core';
import { CommonModule, DecimalPipe } from '@angular/common';
import { DataService, UploadResponse, CleanResponse } from '../../services/data.service'; 
import { FormsModule } from '@angular/forms'; 

@Component({
    selector: 'app-carga-masiva',
    standalone: true, 
    imports: [CommonModule, DecimalPipe, FormsModule], 
    templateUrl: './carga-masiva.html',
    styleUrl: './carga-masiva.css',
})
export class CargaMasiva implements OnInit {

    // 1. ESTADOS CLAVE
    fileToUpload: File | null = null;
    uploading: boolean = false; // Controla el estado "Subiendo..."
    cleaning: boolean = false;
    training: boolean = false;
    dataLoaded: boolean = false; // CLAVE: Indica si hay datos cargados en el backend

    // 2. CONTENIDO Y RESULTADOS DE API
    currentStatus: string = 'Inicia cargando un archivo CSV.';
    uploadResult: UploadResponse | null = null;
    cleanSummary: any | null = null; 
    trainingMetrics: any | null = null; 
    
    // 3. MENSAJES DE ALERTA
    errorMessage: string | null = null;
    successMessage: string | null = null;

    constructor(
        private dataService: DataService,
        private cdr: ChangeDetectorRef // Inyección
    ) {}

    ngOnInit(): void {
        this.checkHealth();
    }

    checkHealth(): void {
        this.dataService.healthCheck().subscribe({
            next: (response) => {
                console.log('Backend OK:', response.message);
                this.currentStatus = 'Servidor conectado. Listo para recibir datos.';
            },
            error: (err) => {
                console.error('Error al conectar con el backend:', err);
                this.errorMessage = 'No se pudo conectar con el servidor (http://localhost:5000).';
                this.currentStatus = 'Servidor desconectado.';
            }
        });
    }

    // Maneja la selección del archivo desde el input
    handleFileInput(event: Event): void {
        const input = event.target as HTMLInputElement;
        if (input.files && input.files.length > 0) {
            this.fileToUpload = input.files[0];
            this.currentStatus = `Archivo listo para subir: ${this.fileToUpload.name}`;
            this.errorMessage = null;
            this.successMessage = null;
            // Asegúrate de resetear los resultados previos al seleccionar un nuevo archivo
            this.resetResults();
        }
    }
    
    // Llama al endpoint POST /api/upload
    uploadFile(): void {
        if (!this.fileToUpload) {
            this.errorMessage = 'Por favor, selecciona un archivo CSV válido.';
            return;
        }
        
        this.uploading = true;
        this.errorMessage = null;
        this.successMessage = null;
        this.currentStatus = 'Subiendo y procesando archivo...';

        this.dataService.uploadFile(this.fileToUpload).subscribe({
            // Manejo de Respuesta Exitosa (HTTP 200 OK)
            next: (response: UploadResponse) => {
                // Diagnóstico en consola:
                console.log('--- RESPUESTA JSON (NEXT/200 OK) ---');
                console.log(JSON.stringify(response, null, 2));
                console.log('------------------------------------');

                this.uploading = false; // Desactivar estado "Subiendo..."
                this.dataLoaded = true; // Activar estado de datos cargados
                this.uploadResult = response;
                
                // Mensaje simplificado solicitado
                this.successMessage = "Datos cargados correctamente y ya"; 
                
                let missingCount = this.getTotalMissingValues();
                this.currentStatus = missingCount > 0 
                    ? `Carga exitosa. Se detectaron ${missingCount} valores faltantes. Listo para limpiar.`
                    : `Carga exitosa. No se detectaron valores faltantes. Listo para limpiar.`;

                this.cleanSummary = null; 
                this.trainingMetrics = null; 
                // 3. FORZAR DETECCIÓN DE CAMBIOS
                this.cdr.detectChanges();
            },
            // Manejo de Errores
            error: (err) => {
                console.error('--- RESPUESTA ERROR JSON/OBJETO ---');
                console.error(err);
                console.log('------------------------------------');
                
                this.uploading = false; // Desactivar estado "Subiendo..."
                
                let errorMsg = 'Error desconocido al subir archivo.';
                if (err.status === 0) {
                    errorMsg = 'Error de Conexión/CORS (Status 0). Asegúrate que Flask esté corriendo y que CORS esté habilitado.';
                } else if (err.error?.error) {
                    errorMsg = err.error.error;
                } else if (err.message) {
                    errorMsg = err.message;
                }

                this.errorMessage = errorMsg;
                this.currentStatus = 'Error durante la carga.';
                this.dataLoaded = false;
            }
        });
    }

    getTotalMissingValues(): number {
        if (!this.uploadResult || !this.uploadResult.info.missing_values) {
            return 0;
        }
        return Object.values(this.uploadResult.info.missing_values).reduce((sum: number, current: any) => sum + current, 0);
    }

    // Llama al endpoint POST /api/clean
  cleanData(): void {
        if (!this.dataLoaded) {
            this.errorMessage = 'No hay datos cargados para limpiar.';
            return;
        }
        
        this.cleaning = true;
        this.errorMessage = null;
        this.successMessage = null;
        this.currentStatus = 'Iniciando limpieza de datos...';

        this.dataService.cleanData().subscribe({
            // MANEJO EXITOSO
            next: (response: CleanResponse) => { // Usamos la interfaz CleanResponse
                this.cleaning = false; // Estado pasa a false
                this.cleanSummary = response;
                
                // Generamos un mensaje de éxito más detallado basado en la respuesta
                const adjusted = response.summary.values_adjusted || 0;
                const message = adjusted > 0
                    ? `Limpieza exitosa. Se ajustaron ${adjusted} valores atípicos.`
                    : 'Limpieza exitosa. No se requirieron ajustes significativos.';
                    
                this.currentStatus = 'Datos limpiados exitosamente. Listo para entrenar.';
                this.successMessage = message;
                this.trainingMetrics = null; 
                
                // CORRECCIÓN CLAVE: Forzar la actualización
                this.cdr.detectChanges(); 
            },
            // MANEJO DE ERROR
            error: (err) => {
                this.cleaning = false; // Estado pasa a false
                const errorMsg = err.error?.error || 'Error desconocido al limpiar datos.';
                this.errorMessage = errorMsg;
                this.currentStatus = 'Error durante la limpieza.';
                
                // CORRECCIÓN CLAVE: Forzar la actualización
                this.cdr.detectChanges(); 
            }
        });
    }

    // Llama al endpoint POST /api/train
trainModel(): void {
        if (!this.cleanSummary) {
            this.errorMessage = 'Primero debes limpiar los datos.';
            return;
        }
        
        this.training = true;
        this.errorMessage = null;
        this.successMessage = null;
        this.currentStatus = 'Iniciando entrenamiento del modelo...';

        this.dataService.trainModel().subscribe({
            // MANEJO EXITOSO
            next: (response) => {
                this.training = false; // Estado pasa a false
                this.trainingMetrics = response.metrics;

                // Generamos un mensaje de éxito más detallado
                const accuracy = response.metrics.accuracy * 100;
                this.currentStatus = `Modelo entrenado y guardado. Accuracy: ${accuracy.toFixed(2)}%.`;
                this.successMessage = response.message;
                
                // CORRECCIÓN CLAVE: Forzar la actualización
                this.cdr.detectChanges(); 
            },
            // MANEJO DE ERROR
            error: (err) => {
                this.training = false; // Estado pasa a false
                const errorMsg = err.error?.error || 'Error desconocido al entrenar modelo.';
                this.errorMessage = errorMsg;
                this.currentStatus = 'Error durante el entrenamiento.';
                
                // CORRECCIÓN CLAVE: Forzar la actualización
                this.cdr.detectChanges(); 
            }
        });
    }
    
    // Limpia todos los estados en el frontend y llama al backend para resetear datos
   resetSystem(): void {
        // Limpiar mensajes de estado antes de la llamada
        this.errorMessage = null;
        this.successMessage = null;
        this.currentStatus = 'Enviando solicitud de reinicio al servidor...';

    this.dataService.resetSystem().subscribe({
      // MANEJO EXITOSO
      next: (response) => {
        this.resetAllStates();
        this.currentStatus = 'Sistema reiniciado. Carga un nuevo archivo.';
        this.successMessage = response.message;
        
                this.cdr.detectChanges(); 
      },
      // MANEJO DE ERROR
      error: (err) => {
        console.error('Error al resetear:', err);
        this.errorMessage = 'Error al intentar reiniciar el sistema. (Revisa la consola del backend).';
        this.currentStatus = 'Error durante el reinicio.';
                
                // CORRECCIÓN CLAVE: Forzar la actualización en caso de error
                this.cdr.detectChanges(); 
      }
    });
  }

    // Función auxiliar para limpiar variables de estado y resultados
    private resetAllStates(): void {
        this.fileToUpload = null;
        this.uploading = false;
        this.cleaning = false;
        this.training = false;
        this.dataLoaded = false;
        this.uploadResult = null;
        this.cleanSummary = null;
        this.trainingMetrics = null;
        this.errorMessage = null;
        this.successMessage = null;
        // Reiniciar el input de archivo si existe
        const fileInput = document.getElementById('file-upload') as HTMLInputElement;
        if (fileInput) {
            fileInput.value = '';
        }
    }
    
    public resetResults(): void {
        this.dataLoaded = false;
        this.uploadResult = null;
        this.cleanSummary = null;
        this.trainingMetrics = null;
        this.currentStatus = `Archivo listo para subir: ${this.fileToUpload?.name}`;
        this.cdr.detectChanges();
    }
}