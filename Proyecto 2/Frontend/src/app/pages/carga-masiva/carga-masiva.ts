import { Component, ChangeDetectorRef } from '@angular/core'; // 1. Importar ChangeDetectorRef
import { DataService, UploadResponse } from '../../services/data.service';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-carga-masiva',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './carga-masiva.html',
  styleUrl: './carga-masiva.css',
})
export class CargaMasiva {
  selectedFile: File | null = null;
  uploadProgress: number = 0;
  statusMessage: string = 'Esperando selección...';
  isUploading: boolean = false;
  lastUploadedFileId: string | null = null;
  history: any[] = [];

  // 2. Inyectar en el constructor
  constructor(
    private dataService: DataService,
    private cdr: ChangeDetectorRef 
  ) {}

  onFileSelected(event: any): void {
    const file = event.target.files[0];
    if (file) {
      this.selectedFile = file;
      this.uploadProgress = 0;
      this.statusMessage = `Listo: ${file.name}`;
      this.cdr.detectChanges(); // Forzar actualización visual al seleccionar
    }
  }

  uploadFile(): void {
    if (!this.selectedFile) return;

    this.isUploading = true;
    this.uploadProgress = 30; // Iniciando
    this.statusMessage = 'Subiendo al servidor...';
    this.cdr.detectChanges(); 

    this.dataService.uploadFile(this.selectedFile).subscribe({
      next: (response: UploadResponse) => {
        console.log('Upload response:', response);
        
        // 3. Actualizar variables tras el 200 OK
        this.uploadProgress = 100;
        this.statusMessage = '¡Archivo cargado correctamente!';
        this.lastUploadedFileId = response.file_id;
        
        this.history.unshift({
          filename: this.selectedFile?.name,
          status: 'success',
          date: new Date()
        });

        this.isUploading = false;
        this.selectedFile = null;
        localStorage.setItem('lastFileId', response.file_id);

        // 4. FORZAR DETECCIÓN DE CAMBIOS
        this.cdr.detectChanges();
      },
      error: (err) => {
        this.isUploading = false;
        this.uploadProgress = 0;
        this.statusMessage = 'Error en la carga';
        this.cdr.detectChanges();
        console.error(err);
      }
    });
  }

  // Método para limpiar todo y volver a empezar
  resetProcess(): void {
    this.selectedFile = null;
    this.uploadProgress = 0;
    this.statusMessage = 'Esperando selección...';
    this.isUploading = false;
    this.cdr.detectChanges();
  }
}